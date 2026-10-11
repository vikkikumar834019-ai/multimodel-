class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,max_seq_len):
      super().__init__()
      self.d_model=d_model
      self.num_heads=num_heads
      self.num_layers=num_layers
      self.max_seq_len=max_seq_len
      self.token_embedding=nn.Embedding(vocab_size,d_model)

      # Decoder step
      # Steps 1
      self.rms_norm1=nn.ModuleList([nn.RMSNorm(d_model) for _ in range(num_layers)])

      # Steps 2
      self.q=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(num_layers)])
      self.k=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(num_layers)])
      self.v=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(num_layers)])
      self.o_proj=nn.ModuleList([nn.Linear(d_model,d_model)for _ in range(num_layers)])
      self.rms_norm2=nn.ModuleList([nn.RMSNorm(d_model) for _ in range(num_layers)])
      hidden_dim = int((8 * d_model) / 3)
      self.swiglu_gate = nn.ModuleList([nn.Linear(d_model, hidden_dim)for _ in range(num_layers)])
      self.swiglu_up = nn.ModuleList([nn.Linear(d_model, hidden_dim)for _ in range(num_layers)])
      self.swiglu_down = nn.ModuleList([nn.Linear(hidden_dim, d_model)for _ in range(num_layers)])
      self.final_norm = nn.RMSNorm(d_model)
      self.lm_head = nn.Linear(d_model, vocab_size)
      self.head_dim=d_model//num_heads
      self.rope_theta = 10000.0
      self.register_buffer("rope_freq",1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float()/ self.head_dim) ))
    def forward(self,x):
      batch_size,seq_len=x.shape
      x=self.token_embedding(x)
      for i in range(self.num_layers):
          x_norm = self.rms_norm1[i](x)
          q = self.q[i](x_norm)
          k = self.k[i](x_norm)
          v = self.v[i](x_norm)

          q = q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
          k = k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
          v = v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)

          positions = torch.arange(seq_len,device=x.device)
          freqs = torch.outer(positions,self.rope_freq)
          cos = torch.cos(freqs)[None, None, :, :]
sin = torch.sin(freqs)[None, None, :, :]
          q_even = q[..., 0::2]
          q_odd = q[..., 1::2]
          q_rotated_even = q_even * cos - q_odd * sin
          q_rotated_odd = q_even * sin + q_odd * cos
          q = torch.stack([q_rotated_even, q_rotated_odd],dim=-1).flatten(-2)
          k_even = k[..., 0::2]
          k_odd = k[..., 1::2]
          k_rotated_even = k_even * cos - k_odd * sin
          k_rotated_odd = k_even * sin + k_odd * cos
          k = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)
          attn_scores = q @ k.transpose(-2, -1)
          attn_scores= attn_scores/(self.head_dim**0.5)
          mask= torch.tril(torch.ones(seq_len,seq_len,device=x.device))
          attn_scores=attn_scores.masked_fill(mask==0,float("-inf"))
          attn_weights=torch.softmax(attn_scores,dim=-1)

          output=attn_weights @ v
          output=output.transpose(1,2)
          output=output.contiguous().view(batch_size,seq_len,self.d_model)
          output = self.o_proj[i](output)
          x = x + output
          x_norm = self.rms_norm2[i](x)
          gate = self.swiglu_gate[i](x_norm)
          up = self.swiglu_up[i](x_norm)
          gate = torch.nn.functional.silu(gate)
          ffn_output = gate * up
          ffn_output = self.swiglu_down[i](ffn_output)
          x = x + ffn_output
      x = self.final_norm(x)
      logits = self.lm_head(x)
      return logits

    @torch.no_grad()
    def gen_response(self,idx,max_new_tokens):
     for i in range(max_new_tokens):
      idx_cond = idx[:, -self.max_seq_len:]
      logits=self(idx_cond)
      logits=logits[:,-1,:]
      next_token=torch.argmax(logits,dim=-1,keepdim=True)
      idx=torch.cat((idx,next_token),dim=1)
     return idx


import torch
import torch.nn as nn


class Text_Train_to_Deploy:

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.avg_train_loss = []
        self.total_train_loss = []
        self.avg_val_loss = []
        self.total_val_loss = []
        self.avg_test_loss = []
        self.total_test_loss = []

    def train(self, model, loader, epoch, loss_fun, optimizer):
        model.to(self.device)

        for _ in range(epoch):
            model.train()
            total_loss = 0.0

            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)

                error = loss_fun(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                error.backward()
                optimizer.step()

                total_loss += error.item()

            avg_loss = total_loss / len(loader)

            self.total_train_loss.append(total_loss)
            self.avg_train_loss.append(avg_loss)

    def validation(self, model, loader, loss_fun):
        model.to(self.device)
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                error = loss_fun(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += error.item()

        avg_loss = total_loss / len(loader)
        self.total_val_loss.append(total_loss)
        self.avg_val_loss.append(avg_loss)

        return avg_loss

    def test(self, model, loader, loss_fun):
        model.to(self.device)
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                error = loss_fun(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += error.item()

        avg_loss = total_loss / len(loader)
        self.total_test_loss.append(total_loss)
        self.avg_test_loss.append(avg_loss)

        return avg_loss

    def deploy(self, model):
        model.to(self.device)
        model.eval()
        return model

    def save(self, model, path):
        torch.save(model.state_dict(), path)

    def load(self, model, path):
        state = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(state)
        model.to(self.device)
        return model

    def generate(
        self, model, tokenizer, prompt, max_length=100
    ):
        model.to(self.device)
        model.eval()

        return model.gen_response(
            prompt,
            tokenizer,
            max_new_tokens=max_length
        )

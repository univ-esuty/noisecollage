import torch
import torchvision
from PIL import Image
import open_clip
import lpips

def batch_per_mse(x, y):
    out = torch.mean((x-y)*(x-y), dim=list(range(1,y.ndim)))
    return [score.item() for score in list(out)]

def batch_per_cosine_similarity(x, y, normalize=True):
    if normalize:
        out = torch.nn.functional.cosine_similarity(x, y, dim=1, eps=1e-8)
    else:
        out = x @ y.T
    return [score.item() for score in list(out)]

def mse(x, y):
    out = torch.mean((x-y)*(x-y))
    return out.item()

def cosine_similarity(x, y, normalize=True):
    if normalize:
        out = torch.nn.functional.cosine_similarity(x, y, dim=1, eps=1e-8)
    else:
        out = x @ y.T
    return out.item()

class EvalCLIP:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='laion400m_e32', device='cuda') -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.device = device
        self.model.to(self.device)
        self.cosine_similality = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def img_preprocess(self, img):
        if isinstance(img, str):
            img = Image.open(img)
            return self.preprocess(img).unsqueeze(0)
        elif isinstance(img, Image.Image):
            return self.preprocess(img).unsqueeze(0)
        elif isinstance(img, torch.Tensor):
            return img
        else:
            raise Exception(f"Invalid image type: {type(img)}")

    def text_preprocess(self, text):
        if isinstance(text, str):
            return self.tokenizer(text)
        elif isinstance(text, torch.Tensor):
            return text
        else:
            raise Exception(f"Invalid image type: {type(text)}")
    
    def loss(self, x, y, batch=1, normalize=True):
        if batch > 1:
            return batch_per_cosine_similarity(x, y, normalize)
        else:
            return cosine_similarity(x, y, normalize)

    torch.no_grad()
    def img_img(self, img_x, img_y, normalize=True, loss=None):
        img_x = self.img_preprocess(img_x).to(self.device)
        img_y = self.img_preprocess(img_y).to(self.device)
        
        x_features = self.model.encode_image(img_x)
        y_features = self.model.encode_image(img_y)

        return self.loss(x_features, y_features, normalize=normalize)

    torch.no_grad()
    def img_text(self, img_x, text_y, normalize=True, loss=None):
        img_x = self.img_preprocess(img_x).to(self.device)
        text_y = self.text_preprocess(text_y).to(self.device)

        x_features = self.model.encode_image(img_x)
        y_features = self.model.encode_text(text_y)

        return self.loss(x_features, y_features, normalize=normalize)

    torch.no_grad()
    def img_features(self, img_x, y_features, normalize=True, loss=None):
        img_x = self.img_preprocess(img_x).to(self.device)
        x_features = self.model.encode_image(img_x)

        return self.loss(x_features, y_features, normalize=normalize)


class EvalLPIPS:
    def __init__(self, net='alex') -> None:
        """
        net (str): The network to use. Currently, 'alex' and 'vgg' are supported.
        """
        self.lpips = lpips.LPIPS(net=net)

    def img_preprocess(self, img):
        tensor = torchvision.transforms.functional.to_tensor(img)
        tensor = (tensor - 0.5) * 2
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def img_img(self, img_x, img_y):
        img_x = self.img_preprocess(img_x)
        img_y = self.img_preprocess(img_y)

        return self.lpips(img_x, img_y).item()


class EvalMSE:
    def __init__(self) -> None:
        pass

    def img_preprocess(self, img):
        return torchvision.transforms.functional.pil_to_tensor(img).to(torch.float32) / 255.0
    
    @torch.no_grad()
    def img_img(self, img_x, img_y, **kwargs):
        img_x = self.img_preprocess(img_x)
        img_y = self.img_preprocess(img_y)

        if kwargs.get('loss', None) is not None:
            raise Exception("[Warning] You try to use the other loss method instaed of MSE in EvalMSE.")
        else:
            if kwargs.get('batch', 1) > 1:
                return batch_per_mse(img_x, img_y)
            else:
                return mse(img_x, img_y)
            

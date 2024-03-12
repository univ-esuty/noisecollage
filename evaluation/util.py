from PIL import Image

def squarize(pil_img, fill):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), fill)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), fill)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def trim_by_mask(img:Image.Image, mask_img:Image.Image, out_res=512, fill=255):
    """
    img: Image to be trimmed
    mask_img: Image of mask
    out_res: Resolution of output image
    fill: Fill value of output image
    """
    ## Get Bounding Box
    bbox = mask_img.getbbox()
    if bbox is None:
        raise Exception("Invalid mask image")
    
    ## Crop Image
    img = img.crop(bbox)
    img = squarize(img, fill)
    mask_img = mask_img.crop(bbox)
    mask_img = squarize(mask_img, 0)
    
    ## Resize
    img = img.resize((out_res, out_res))
    mask_img = mask_img.resize((out_res, out_res))
    
    ## Fill
    img = Image.composite(img, Image.new("RGB", (out_res, out_res), color=(fill,fill,fill)), mask_img)
    
    return img
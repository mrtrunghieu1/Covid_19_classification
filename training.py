'''Main function to '''
import timm

def create_model(flag_model, flag_pretrained=True):
    
    if flag_model == 0:
        model = timm.create_model('efficientnet_b0', pretrained=flag_pretrained)
    elif flag_model == 1:
        model = timm.create_model('efficientnet_b1', pretrained=flag_pretrained)
    elif flag_model == 2:
        model = timm.create_model('efficientnet_b2', pretrained=flag_pretrained)
    elif flag_model == 3:
        model = timm.create_model('efficientnet_b3', pretrained=flag_pretrained)
    elif flag_model == 4:
        model = timm.create_model('mixnet_s', pretrained=flag_pretrained)
    elif flag_model == 5:
        model = timm.create_model('mixnet_m', pretrained=flag_pretrained)
    elif flag_model == 6:
        model = timm.create_model('mixnet_l', pretrained=flag_pretrained)
    else:
        raise Exception("Not models pretrained!!!")

    return model

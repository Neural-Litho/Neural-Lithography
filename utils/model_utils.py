
from config import *
from net.fno import FNO2d
from model.learned_litho import LearnedLitho3D, LearnedLitho3D2



def model_selector(model_choice):
        
    if model_choice == 'physics':
        model = LearnedLitho3D(dx=0.1, sigmao=0.2, sigmac=2.5, thresh=0.5, kmax=1.0, kmin=0.5).to(device)
    
    elif model_choice == 'pbl3d':
        model = LearnedLitho3D2(dx=0.1, sigmao=0.2, sigmac=2.5).to(device)
    
    elif model_choice == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=12).to(device)
        
    else:
        raise NotImplementedError
    
    return model

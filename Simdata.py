
import numpy as np
import random
def siminput(shape):
    
    eles = []
    
    h, w, c =shape[0], shape[1], shape[2]
    
    for px in range(h * w * c):
        
        eles.append(random.randint(0, 255))
        
    data = np.array(eles)
    
    simx = np.reshape(data, shape).astype(np.uint8)
    
    return simx

def main():
    
    data = siminput([14, 14, 3])
    
    print(data.shape)
    
    dic = {'pa': 50, 'miou': 60}
    
    maxpa = {'pa': 40, 'miou': 0}
    
    maxiou = {'pa': 30, 'miou': 40}
    
    if maxpa['pa'] <= dic['pa']:
        
        maxpa = dic
        
        print(maxpa)
        
    if maxiou['miou'] <= dic['miou']:
        
        maxiou = dic
    
        print(maxiou)
        
    
if __name__ == '__main__':
    
    main()
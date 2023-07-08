import os
import shutil
import json
import pdb
sorce_train_path='/data/imagenet/train'
sorce_val_path='/data/imagenet/val'
target_train_path='/data/imagenet100/train'
target_val_path='/data/imagenet100/val'
json_path='./imagenet_class_index.json'

#json read

pdb.set_trace()
with open(json_path,'r') as f:
    data=json.load(f)
    

ima_1k = data
ima_100 = [
        'Robin', 'Gila monster', 'hognose snake', 'garter snake', 'green mamba', 
        'garden spider', 'lorikeet', 'goose', 'rock crab', 'fiddler crab', 'American lobster', 
        'little blue heron', 'American coot', 'Chihuahua', 'Shih Tzu', 'Papillon', 'toy terrier', 
        'Walker hound', 'English foxhound', 'borzoi', 'Saluki', 'American Staffordshire Terrier', 
        'Chesapeake Bay Retriever', 'Vizsla', 'Kuvasz', 'Komondor', 'Rottweiler', 'Doberman', 'Boxer', 
        'Great Dane', 'Standard Poodle', 'Mexican hairless', 'coyote', 'African hunting dog', 
        'red fox','tabby', 'meerkat', 'dung beetle', 'walking stick', 'leafhopper', 'hare', 'wild boar', 
        'gibbon', 'langur', 'ambulance', 'bannister', 'bassinet', 'boathouse', 'bonnet', 
        'bottlecap', 'car wheel', 'chime', 'cinema', 'cocktail shaker', 'computer keyboard', 
        'Dutch oven', 'football helmet', 'gasmask', 'hard disc', 'harmonica', 'honeycomb', 
        'iron', 'jean', 'lampshade', 'laptop', 'milk can', 'mixing bowl', 'modem', 'moped', 
        'shower cap', 'mousetrap', 'obelisk', 'park bench', 'pedestal', 'pickup', 'pirate', 
        'purse', 'reel', 'rocking chair', 'rotisserie', 'safety pin', 'sarong', 'ski mask', 'slide rule', 'stretcher', 'theater curtain', 'throne', 'tile roof', 'tripod', 'tub', 'vacuum', 'window screen', 'wing', 'head cabbage', 'cauliflower', 'pineapple', 'carbonara', 'chocolate sauce', 'gyromitra', 'mushroom']

for i in range(len(ima_100)):
    ima_100[i]=ima_100[i].lower()
length=0
pdb.set_trace()
for file in data.values():
    file[1]=file[1].replace('_',' ')
    file[1]=file[1].replace('-',' ')
    if file[1].lower() in ima_100:
        ima_100.remove(file[1].lower())
        length+=1
        try:
            shutil.copytree(os.path.join(sorce_train_path,file[0]),os.path.join(target_train_path,file[0]))
            shutil.copytree(os.path.join(sorce_val_path,file[0]),os.path.join(target_val_path,file[0]))
            print('copy',file[1])
        except:
            print('error',file[1])
print(ima_100)
print(length)
pdb.set_trace()
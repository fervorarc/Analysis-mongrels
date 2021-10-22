import sys
import os
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


originals_path = '~/testTriples/88triples/'
mongrel_path = '~/88out/'

img2vec = Img2Vec()

simList = []
# For each test image, we store the filename and vector as key, value in a dictionary
for og, mg in zip(sorted(glob.glob(originals_path+"*.png")), sorted(glob.glob(mongrel+path""))):
    ogi = Image.open(og)
    mgi = Image.open(mg)
    print(ogi.shape)
    ogVec = img2vec.get_vec(ogi)
    mgVec = img2vec.get_vec(mgi)
    cs = cosine_similarity(ogVec.reshape((1, -1)), mgVec.reshape((1, -1)))[0][0]
    simList.append((og, cs))
    
print(sorted(simList))
    
pics = {}
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    pics[filename] = vec

# pic_name = ""
# while pic_name != "exit":
#     pic_name = str(input("Which filename would you like similarities for?\n"))

#     try:
#         sims = {}
#         for key in list(pics.keys()):
#             if key == pic_name:
#                 continue

#             sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

#         d_view = [(v, k) for k, v in sims.items()]
#         d_view.sort(reverse=True)
#         for v, k in d_view:
#             print(v, k)

#     except KeyError as e:
#         print('Could not find filename %s' % e)

#     except Exception as e:
#         print(e)

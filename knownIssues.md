# Auto Annotate - BR - Bugs Relatados

Compartilhamos abaixo alguns dos problemas conhecidos da ferramenta Auto-Annotate-BR. Esses 'bugs' são basicamente os que surgem devido à disparidade do ambiente python (versão diferente dos pacotes instalados localmente - ver requirements.txt) ou aos bugs que ainda não foram resolvidos e têm uma solução alternativa.

**❤ </> Fique à vontade para contribuir para esta página. </> ❤**

---
## 1. Mensagem de erro: AttributeError: 'str' object has no attribute 'decode'

### Colaborador     :  [jarleven](https://github.com/jarleven)
### Solução:
Downgrade h5py. 
```
pip install 'h5py==2.10.0' --force-reinstall
```
### Referência:
https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi

### Stack Trace:
python3 annotate.py annotateCustom --image_directory=/host/Notes/SalmonModel --label=Salmon --weights=/host/mask_rcnn_coco_tf22_01082021/mask_rcnn_mmodel_0030.h5 --displayMaskedImages=True

```
Loading weights... /host/mask_rcnn_coco_tf22_01082021/mask_rcnn_mmodel_0030.h5
Traceback (most recent call last):
File "annotate.py", line 218, in
model.load_weights(weights_path, by_name=True)
File "/Auto-Annotate/mrcnn/model.py", line 2130, in load_weights
saving.load_weights_from_hdf5_group_by_name(f, layers)
File "/usr/local/lib/python3.6/dist-packages/keras/engine/topology.py", line 3114, in load_weights_from_hdf5_group_by_name
original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'
```
---

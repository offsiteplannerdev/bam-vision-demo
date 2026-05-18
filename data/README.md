# Data Directory

Real manufacturing images should stay out of version control unless they have been explicitly approved for sharing.

The inspection images come from three industrial cameras: `CAM_01`, `CAM_02`, and `CAM_03`.

Annotations are exported in CVAT XML 1.1 format.

The current dataset is imbalanced, with approximately 85% `ok` samples and 15% defect samples.

## TODO

- [ ] Convertir CVAT XML → YOLO txt
- [ ] Split train/val/test estratificado
- [ ] Augmentar clases minoritarias
- [ ] Auditar imágenes corruptas y duplicadas
- [ ] Generar informe de estadísticas del dataset

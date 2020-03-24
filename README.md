# Description
This is a kaggle competition, you can see it through this [Steel defect detection](https://www.kaggle.com/c/severstal-steel-defect-detection). The datasets of this competition you can download form the above website<br><br>
Steel is one of the most important building materials of modern times. Steel buildings are resistant to natural and man-made wear which has made the material ubiquitous around the world. To help make production of steel more efficient, this competition will help identify defects.<br><br>
Severstal is leading the charge in efficient steel mining and production. They believe the future of metallurgy requires development across the economic, ecological, and social aspects of the industry—and they take corporate responsibility seriously. The company recently created the country’s largest industrial data lake, with petabytes of data that were previously discarded. Severstal is now looking to machine learning to improve automation, increase efficiency, and maintain high quality in their production.<br><br>
The production process of flat sheet steel is especially delicate. From heating and rolling, to drying and cutting, several machines touch flat steel by the time it’s ready to ship. Today, Severstal uses images from high frequency cameras to power a defect detection algorithm.<br><br>
In this competition, you’ll help engineers improve the algorithm by localizing and classifying surface defects on a steel sheet.<br><br>
If successful, you’ll help keep manufacturing standards for steel high and enable Severstal to continue their innovation, leading to a stronger, more efficient world all around us.<br><br>
# Dependencies
pytorch <br>
jupyter <br>

# Steps
1.The eda of this datasets.You can have a general idea of this datasets. The code in the file of steel-defect-detection-eda.ipynb <br>
2.The baseline of segmentation model.In this competition we are not only classification but alse segmentation .The code in the file of steel-defect-detection-segmentation <br>
3.We use model ensemble whice include se_resnext50_32x4d、mobilenet2 and resnet34 to build the segmentation modle. The code in the file of steel-defect-detection-ensemble <br>
4.The classifier of this datasets <br>
5.The whole model of this competition.we use segmentation and classification to build this model.The code in the file of steel-defect-detection-final <br>

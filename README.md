### Pytorch deployment - Image classification using Resnet-18 pretrained on Imagenet Dataset

### Inspired from https://github.com/imadtoubal/Pytorch-Flask-Starter

#### Description
Deploy our PyTorch model with Flask and Heroku and Docker. Create a simple Flask app with a REST API that returns the result as json data, create a docker image and then deploy it to Heroku. 
Here we will do image classification, and we can send images to our heroku app and then predict it with our live running app. You can upload multiple images for simultaneous inferencing. If uploaded multiple images inferencing will be done in batches of the number of upladed images. Used streamer to handle multiple inferencing requests

#### Heroku App Link :
https://pfdwebapp.herokuapp.com/

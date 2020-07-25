nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 -v /home/rizawa/Codes/recruitement/assignment:/home/rizawa/Codes/recruitement/assignment --name homework nvcr.io/nvidia/pytorch:19.09-py3 

tensorboard:
docker run --runtime=nvidia -it -p 6007:6007 -v /home/rizawa/Codes/recruitement/assignment:/home/rizawa/Codes/recruitement/assignment tensorflow/tensorflow:latest

tensorboard --port 6007 --logdir /home/rizawa/Codes/recruitement/assignment/SimpleCNN/logs --host=0.0.0.0
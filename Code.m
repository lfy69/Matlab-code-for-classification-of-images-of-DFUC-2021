clear;clc;close;

% file_path='F:\lianxi_M\d\'; %file path
% files=dir(strcat(file_path,'*.jpg'));
% num=length(files);
% N4=[];
% for i=1:num
%     file_name=files(i).name
%     a1=imread(strcat(file_path,file_name));
%     a2=rgb2gray(a1);
%     a3=im2double(a2);
%    % N=cat(1,N,a3);
%    N4=[N4;a3];
% end

load ('a.mat');% a,b,c,and are pre-processed image data
load ('b.mat');
load ('c.mat');
load ('d.mat');
image=[N1;N2;N3;N4];
% 制备标签
lab1=ones(2551,1);
lab2=2*ones(2554,1);
lab3=3*ones(226,1);
lab4=4*ones(620,1);
label=[lab1;lab2;lab3;lab4];

% for i=1:1:32
%     subplot(4,8,i);
%     imshow(image(:,:,i))
% end

X = reshape(image, [224,224,1,5951]); % 5951 images
                                      
size(X)                              
Y = categorical(label);              
idx = randperm(5951);   
num_train = round(0.8*length(X)); 
num_val = round(0.1*length(X));  
% image data
X_train = X(:,:,:,idx(1:num_train));
X_val = X(:,:,:,idx(num_train+1:num_train+num_val));
X_test = X(:,:,:,idx(num_train+num_val+1:end));  

% labels 
Y_train = Y(idx(1:num_train),:);
Y_val = Y(idx(num_train+1:num_train+num_val),:);
Y_test = Y(idx(num_train+num_val+1:end),:);

% define the network layers
layers = [...
          imageInputLayer([224,224,1]);
          batchNormalizationLayer();  
          
          convolution2dLayer(3,16);  
          batchNormalizationLayer();
          reluLayer()                
         % dropoutLayer
          maxPooling2dLayer(2,'Stride',2);
          
          convolution2dLayer(3,32);  
          batchNormalizationLayer();
          reluLayer()                 
         % dropoutLayer(0.3)                
          maxPooling2dLayer(2,'Stride',2); 
                              
          fullyConnectedLayer(4);       
          softmaxLayer();                
          classificationLayer(),...
    ];

% 参数（存在验证集）
options = trainingOptions('sgdm',...                         
                          'MiniBatchSize',64, ...
                          'MaxEpochs',12,...                 
                          'ValidationData',{X_val,Y_val},... 
                          'Verbose',true, ...                
                          'Shuffle','every-epoch', ...
                          'InitialLearnRate',0.001,...
                          'Plots','training-progress');
net_cnn = trainNetwork(X_train,Y_train,layers,options);

testLabel = classify(net_cnn,X_test);
%[class,err]=classify(net_cnn,X_test);
precision = sum(testLabel==Y_test)/numel(testLabel);
disp(['acuracy',num2str(precision*100),'%'])
plotconfusion(Y_test,testLabel)

%=====================================
file_path='F:\lianxi_M\Val2021\';
files=dir(strcat(file_path,'*.jpg'));
num=length(files);
N=[];
for i=1:num
    file_name=files(i).name
    a1=imread(strcat(file_path,file_name));
    a2=rgb2gray(a1);
    a3=im2double(a2);
   % N=cat(1,N,a3);
   N=[N;a3];
end
X2 = reshape(N, [224,224,1,500]);% 500 images
[class,err] = classify(net_cnn,X2);
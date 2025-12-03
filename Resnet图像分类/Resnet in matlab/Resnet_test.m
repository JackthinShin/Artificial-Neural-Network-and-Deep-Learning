%% 加载数据集
% Train Deep Learning Network to Classify New Images

% Load Data
digitDatasetPath = fullfile('E:\matlab\aaaworkspace\Resnet in matlab', '/MerchData/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

net = resnet18;

% analyzeNetwork(net)

% The first element of the Layers property of the network is the image input layer. 
net.Layers(1)
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 


[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

% In most networks, the last layer with learnable weights is a fully connected layer. 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

% Freeze Initial Layers

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% Train Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% To automatically resize the validation images without performing further data augmentation, use an augmented image datastore without specifying any additional preprocessing operations.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Specify the training options. Set InitialLearnRate to a small value to slow down learning in the transferred layers that are not already frozen. In the previous step, you increased the learning rate factors for the last learnable layer to speed up learning in the new final layers. This combination of learning rate settings results in fast learning in the new layers, slower learning in the middle layers, and no learning in the earlier, frozen layers. 

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network using the training data. By default, trainNetwork uses a GPU if one is available (requires Parallel Computing Toolbox? and a CUDA? enabled GPU with compute capability 3.0 or higher). Otherwise, trainNetwork uses a CPU. You can also specify the execution environment by using the 'ExecutionEnvironment' name-value pair argument of trainingOptions. Because the data set is so small, training is fast.
net = trainNetwork(augimdsTrain,lgraph,options)

% Classify Validation Images
% Classify the validation images using the fine-tuned network, and calculate the classification accuracy.
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
trainError = mean(YPred ~= imdsValidation.Labels)

% Display four sample validation images with predicted labels and the predicted probabilities of the images having those labels.
idx = randperm(numel(imdsValidation.Files),6);
figure
for i = 1:6
    subplot(2,3,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

% 可视化混淆矩阵
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
ccscat = confusionchart(categorical(imdsValidation.Labels) ,YPred);
ccscat.Title = '混淆矩阵';
ccscat.ColumnSummary = 'column-normalized';
ccscat.RowSummary = 'row-normalized';




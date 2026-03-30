function results = two_stage_four_class_classifier(varargin)
% Two-stage four-class classifier:
% Stage 1: Normal vs Abnormal
% Stage 2: Bleeding vs Emptying vs Obstruction on abnormal clips only

opts = parseInputs(varargin{:});

dataFolder = fullfile(pwd, "出血-排空-梗阻-正常");
outputFolder = opts.outputFolder;

if ~isfolder(dataFolder)
    error("Data folder not found: %s", dataFolder);
end
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

fileList = dir(fullfile(dataFolder, "*.csv"));
[X, Y, groups, clipTable, classNames, sequenceLength] = buildDataset( ...
    fileList, opts.windowSeconds, opts.stepSeconds, opts.maxClipsPerFile, opts.randomSeed, opts.useDiffChannels);

repeatAccuracies = zeros(opts.numRepeats, 1);
repeatConfusions = zeros(numel(classNames), numel(classNames), opts.numRepeats);
stage1Accuracies = zeros(opts.numRepeats, 1);
stage2Accuracies = zeros(opts.numRepeats, 1);
stage1Confusions = zeros(2, 2, opts.numRepeats);
stage2Confusions = zeros(3, 3, opts.numRepeats);
predictionTables = cell(opts.numRepeats, 1);
repeatRows = {};

for repeatIdx = 1:opts.numRepeats
    currentSeed = opts.randomSeed + repeatIdx - 1;
    [trainIdx, testIdx] = stratifiedSplit(Y, opts.trainRatio, currentSeed);

    XTrain = X(trainIdx);
    YTrain = categorical(Y(trainIdx), classNames, classNames);
    XTest = X(testIdx);
    YTest = categorical(Y(testIdx), classNames, classNames);
    groupsTest = groups(testIdx);

    [XTrain, XTest] = normalizeByTrainingSet(XTrain, XTest);
    [XTrain, YTrain] = augmentTrainingSet(XTrain, YTrain, ...
        opts.augmentationCopies, opts.noiseStd, opts.scaleJitter, ...
        opts.timeShiftMax, opts.maskFraction, currentSeed);

    [predictedLabels, stage1Pred, stage2Pred, stage2True] = classifyTwoStage(XTrain, YTrain, XTest, YTest, classNames, sequenceLength, opts);

    repeatConfusions(:, :, repeatIdx) = confusionmat(YTest, predictedLabels, ...
        'Order', categorical(classNames, classNames, classNames));
    repeatAccuracies(repeatIdx) = mean(YTest == predictedLabels);
    YTestStage1 = repmat("异常", size(YTest));
    YTestStage1(YTest == "正常") = "正常";
    YTestStage1 = categorical(YTestStage1, ["异常","正常"], ["异常","正常"]);
    stage1Confusions(:, :, repeatIdx) = confusionmat(YTestStage1, stage1Pred, ...
        'Order', categorical(["异常","正常"], ["异常","正常"], ["异常","正常"]));
    stage1Accuracies(repeatIdx) = mean(YTestStage1 == stage1Pred);
    if ~isempty(stage2True)
        stage2TrueFixed = categorical(string(stage2True), ["出血","排空","梗阻"], ["出血","排空","梗阻"]);
        stage2PredFixed = categorical(string(stage2Pred), ["出血","排空","梗阻"], ["出血","排空","梗阻"]);
        stage2Confusions(:, :, repeatIdx) = confusionmat(stage2TrueFixed, stage2PredFixed, ...
            'Order', categorical(["出血","排空","梗阻"], ["出血","排空","梗阻"], ["出血","排空","梗阻"]));
        stage2Accuracies(repeatIdx) = mean(stage2TrueFixed == stage2PredFixed);
    else
        stage2Accuracies(repeatIdx) = NaN;
    end
    predictionTables{repeatIdx} = table( ...
        repmat(repeatIdx, numel(YTest), 1), string(YTest), string(predictedLabels), groupsTest, ...
        'VariableNames', ["Repeat", "TrueLabel", "PredictedLabel", "SourceFile"]);
    repeatRows(end + 1, :) = {repeatIdx, currentSeed, repeatAccuracies(repeatIdx), stage1Accuracies(repeatIdx), stage2Accuracies(repeatIdx), sum(trainIdx), sum(testIdx)}; %#ok<AGROW>
    fprintf("Repeat %d/%d, seed=%d, overall=%.4f, stage1=%.4f, stage2=%.4f\n", ...
        repeatIdx, opts.numRepeats, currentSeed, repeatAccuracies(repeatIdx), stage1Accuracies(repeatIdx), stage2Accuracies(repeatIdx));
end

meanConfusion = mean(repeatConfusions, 3);
meanAccuracy = mean(repeatAccuracies);
stdAccuracy = std(repeatAccuracies);
meanStage1Confusion = mean(stage1Confusions, 3);
meanStage2Confusion = mean(stage2Confusions, 3);
meanStage1Accuracy = mean(stage1Accuracies);
stdStage1Accuracy = std(stage1Accuracies);
validStage2Acc = stage2Accuracies(~isnan(stage2Accuracies));
meanStage2Accuracy = mean(validStage2Acc);
stdStage2Accuracy = std(validStage2Acc);
predictionTable = vertcat(predictionTables{:});
repeatSummary = cell2table(repeatRows, ...
    'VariableNames', {'Repeat', 'Seed', 'SegmentAccuracy', 'Stage1Accuracy', 'Stage2Accuracy', 'TrainClips', 'TestClips'});

saveConfusionChart(meanConfusion, classNames, ...
    sprintf("Two-stage average confusion matrix (accuracy %.2f%% +/- %.2f%%)", ...
    meanAccuracy * 100, stdAccuracy * 100), ...
    fullfile(outputFolder, "two_stage_segment_confusion_matrix.png"));
saveConfusionChart(meanStage1Confusion, ["异常","正常"], ...
    sprintf("Stage 1 average confusion matrix (accuracy %.2f%% +/- %.2f%%)", ...
    meanStage1Accuracy * 100, stdStage1Accuracy * 100), ...
    fullfile(outputFolder, "two_stage_stage1_confusion_matrix.png"));
saveConfusionChart(meanStage2Confusion, ["出血","排空","梗阻"], ...
    sprintf("Stage 2 average confusion matrix (accuracy %.2f%% +/- %.2f%%)", ...
    meanStage2Accuracy * 100, stdStage2Accuracy * 100), ...
    fullfile(outputFolder, "two_stage_stage2_confusion_matrix.png"));

writetable(clipTable, fullfile(outputFolder, "two_stage_clip_metadata.csv"), 'Encoding', 'UTF-8');
writetable(predictionTable, fullfile(outputFolder, "two_stage_clip_predictions.csv"), 'Encoding', 'UTF-8');
writetable(repeatSummary, fullfile(outputFolder, "two_stage_repeat_summary.csv"), 'Encoding', 'UTF-8');

results = struct();
results.windowSeconds = opts.windowSeconds;
results.stepSeconds = opts.stepSeconds;
results.sequenceLength = sequenceLength;
results.numRepeats = opts.numRepeats;
results.segmentAccuracy = meanAccuracy;
results.segmentAccuracyStd = stdAccuracy;
results.repeatAccuracies = repeatAccuracies;
results.segmentConfusion = meanConfusion;
results.repeatConfusions = repeatConfusions;
results.stage1Accuracy = meanStage1Accuracy;
results.stage1AccuracyStd = stdStage1Accuracy;
results.stage1Accuracies = stage1Accuracies;
results.stage1Confusion = meanStage1Confusion;
results.stage1Confusions = stage1Confusions;
results.stage2Accuracy = meanStage2Accuracy;
results.stage2AccuracyStd = stdStage2Accuracy;
results.stage2Accuracies = stage2Accuracies;
results.stage2Confusion = meanStage2Confusion;
results.stage2Confusions = stage2Confusions;
results.classNames = classNames;
results.clipMetadata = clipTable;
results.predictionTable = predictionTable;
results.repeatSummary = repeatSummary;
results.options = opts;

save(fullfile(outputFolder, "two_stage_classification_results.mat"), "results");

disp("Two-stage mean segment confusion matrix:");
disp(array2table(meanConfusion, 'VariableNames', cellstr(classNames), 'RowNames', cellstr(classNames)));
fprintf("Two-stage mean segment accuracy: %.4f +/- %.4f\n", meanAccuracy, stdAccuracy);
disp("Stage 1 mean confusion matrix:");
disp(array2table(meanStage1Confusion, 'VariableNames', {'异常','正常'}, 'RowNames', {'异常','正常'}));
fprintf("Stage 1 mean accuracy: %.4f +/- %.4f\n", meanStage1Accuracy, stdStage1Accuracy);
disp("Stage 2 mean confusion matrix:");
disp(array2table(meanStage2Confusion, 'VariableNames', {'出血','排空','梗阻'}, 'RowNames', {'出血','排空','梗阻'}));
fprintf("Stage 2 mean accuracy: %.4f +/- %.4f\n", meanStage2Accuracy, stdStage2Accuracy);
fprintf("Outputs saved in: %s\n", outputFolder);
end

function opts = parseInputs(varargin)
parser = inputParser;
addParameter(parser, 'outputFolder', fullfile(pwd, "TwoStage四分类结果"));
addParameter(parser, 'windowSeconds', 1.2);
addParameter(parser, 'stepSeconds', 0.6);
addParameter(parser, 'maxClipsPerFile', 0);
addParameter(parser, 'randomSeed', 42);
addParameter(parser, 'trainRatio', 0.8);
addParameter(parser, 'numRepeats', 5);
addParameter(parser, 'useDiffChannels', true);
addParameter(parser, 'augmentationCopies', 1);
addParameter(parser, 'noiseStd', 0.01);
addParameter(parser, 'scaleJitter', 0.05);
addParameter(parser, 'timeShiftMax', 3);
addParameter(parser, 'maskFraction', 0.05);

% Stage 1: normal vs abnormal
addParameter(parser, 'stage1MiniBatchSize', 32);
addParameter(parser, 'stage1MaxEpochs', 18);
addParameter(parser, 'stage1InitialLearnRate', 6e-4);
addParameter(parser, 'stage1ValidationPatience', 6);
addParameter(parser, 'stage1EmbedDim', 64);
addParameter(parser, 'stage1NumHeads', 4);
addParameter(parser, 'stage1KeyDim', 16);
addParameter(parser, 'stage1DropoutProbability', 0.15);
addParameter(parser, 'stage1ClassWeights', [1.3 1.3]); % abnormal, normal

% Stage 2: bleeding/emptying/obstruction
addParameter(parser, 'stage2MiniBatchSize', 32);
addParameter(parser, 'stage2MaxEpochs', 20);
addParameter(parser, 'stage2InitialLearnRate', 7e-4);
addParameter(parser, 'stage2ValidationPatience', 6);
addParameter(parser, 'stage2EmbedDim', 96);
addParameter(parser, 'stage2NumHeads', 4);
addParameter(parser, 'stage2KeyDim', 24);
addParameter(parser, 'stage2DropoutProbability', 0.18);
addParameter(parser, 'stage2ClassWeights', [1.2 1 1]); % bleeding, emptying, obstruction
parse(parser, varargin{:});
opts = parser.Results;
end

function [predictedLabels, stage1Pred, stage2Pred, stage2True] = classifyTwoStage(XTrain, YTrain, XTest, YTest, classNames, sequenceLength, opts)
stage1Names = ["异常", "正常"];
stage2Names = ["出血", "排空", "梗阻"];

YTrainStage1 = repmat("异常", size(YTrain));
YTrainStage1(YTrain == "正常") = "正常";
YTrainStage1 = categorical(YTrainStage1, stage1Names, stage1Names);

YTestStage1 = repmat("异常", size(YTest));
YTestStage1(YTest == "正常") = "正常";
YTestStage1 = categorical(YTestStage1, stage1Names, stage1Names);

stage1Layers = createNetwork(size(XTrain{1}, 1), sequenceLength, stage1Names, ...
    opts.stage1EmbedDim, opts.stage1NumHeads, opts.stage1KeyDim, ...
    opts.stage1DropoutProbability, opts.stage1ClassWeights);
stage1ValFreq = max(1, floor(numel(XTrain) / opts.stage1MiniBatchSize));
stage1Options = createTrainingOptions(XTest, YTestStage1, ...
    opts.stage1MiniBatchSize, opts.stage1MaxEpochs, opts.stage1InitialLearnRate, ...
    opts.stage1ValidationPatience, stage1ValFreq);
stage1Net = trainNetwork(XTrain, YTrainStage1, stage1Layers, stage1Options);
stage1Pred = classify(stage1Net, XTest, MiniBatchSize=opts.stage1MiniBatchSize);

abnormalTrainMask = YTrain ~= "正常";
abnormalTestMask = stage1Pred ~= "正常";

stage2Layers = createNetwork(size(XTrain{1}, 1), sequenceLength, stage2Names, ...
    opts.stage2EmbedDim, opts.stage2NumHeads, opts.stage2KeyDim, ...
    opts.stage2DropoutProbability, opts.stage2ClassWeights);

YTrainStage2 = removecats(YTrain(abnormalTrainMask));
if any(YTest ~= "正常")
    YValStage2 = removecats(YTest(YTest ~= "正常"));
    XValStage2 = XTest(YTest ~= "正常");
else
    YValStage2 = YTrainStage2;
    XValStage2 = XTrain(abnormalTrainMask);
end
stage2ValFreq = max(1, floor(sum(abnormalTrainMask) / opts.stage2MiniBatchSize));
stage2Options = createTrainingOptions(XValStage2, YValStage2, ...
    opts.stage2MiniBatchSize, opts.stage2MaxEpochs, opts.stage2InitialLearnRate, ...
    opts.stage2ValidationPatience, stage2ValFreq);
stage2Net = trainNetwork(XTrain(abnormalTrainMask), YTrainStage2, stage2Layers, stage2Options);

predictedLabels = repmat(categorical("正常", classNames, classNames), numel(XTest), 1);
stage2Pred = categorical(strings(0,1), stage2Names, stage2Names);
stage2True = categorical(strings(0,1), stage2Names, stage2Names);
if any(abnormalTestMask)
    stage2Pred = classify(stage2Net, XTest(abnormalTestMask), MiniBatchSize=opts.stage2MiniBatchSize);
    stage2True = removecats(YTest(abnormalTestMask));
    predictedLabels(abnormalTestMask) = categorical(string(stage2Pred), classNames, classNames);
end
end

function [X, Y, groups, clipTable, classNames, sequenceLength] = buildDataset(fileList, windowSeconds, stepSeconds, maxClipsPerFile, randomSeed, useDiffChannels)
classNames = ["出血", "排空", "梗阻", "正常"];
X = {};
Y = strings(0, 1);
groups = strings(0, 1);
clipRows = {};
clipCounter = 0;
sequenceLength = [];

for k = 1:numel(fileList)
    sourceFile = fileList(k).name;
    label = inferLabel(sourceFile);
    if label == ""
        continue;
    end

    raw = readmatrix(fullfile(fileList(k).folder, sourceFile));
    timeUs = raw(:, 1);
    signal = single(raw(:, 2:end));

    dtSeconds = median(diff(timeUs)) / 1e6;
    windowRows = max(8, round(windowSeconds / dtSeconds));
    stepRows = max(1, round(stepSeconds / dtSeconds));
    if isempty(sequenceLength)
        sequenceLength = windowRows;
    end

    startRows = 1:stepRows:(size(signal, 1) - windowRows + 1);
    if maxClipsPerFile > 0 && numel(startRows) > maxClipsPerFile
        rng(randomSeed + k, "twister");
        startRows = sort(startRows(randperm(numel(startRows), maxClipsPerFile)));
    end

    segmentIndex = 0;
    for startRow = startRows
        segmentIndex = segmentIndex + 1;
        clipCounter = clipCounter + 1;
        idx = startRow:(startRow + windowRows - 1);
        clipSignal = signal(idx, :)';
        if useDiffChannels
            diffSignal = [zeros(size(clipSignal, 1), 1, 'like', clipSignal), diff(clipSignal, 1, 2)];
            clipSignal = [clipSignal; diffSignal];
        end

        X{clipCounter, 1} = clipSignal; %#ok<AGROW>
        Y(clipCounter, 1) = label; %#ok<AGROW>
        groups(clipCounter, 1) = string(sourceFile); %#ok<AGROW>
        clipRows(end + 1, :) = {sourceFile, label, segmentIndex, idx(1), idx(end), timeUs(idx(1)) / 1e6, timeUs(idx(end)) / 1e6}; %#ok<AGROW>
    end
end

clipTable = cell2table(clipRows, ...
    'VariableNames', {'SourceFile', 'Label', 'ClipIndex', 'StartRow', 'EndRow', 'StartTimeSeconds', 'EndTimeSeconds'});
end

function [trainIdx, testIdx] = stratifiedSplit(labels, trainRatio, randomSeed)
labels = string(labels);
labelNames = unique(labels, "stable");
trainIdx = false(size(labels));
testIdx = false(size(labels));

for i = 1:numel(labelNames)
    mask = labels == labelNames(i);
    idx = find(mask);
    rng(randomSeed + i, "twister");
    idx = idx(randperm(numel(idx)));
    nTrain = max(1, floor(trainRatio * numel(idx)));
    trainIdx(idx(1:nTrain)) = true;
    testIdx(idx(nTrain + 1:end)) = true;
end
end

function [XTrain, XTest] = normalizeByTrainingSet(XTrain, XTest)
allTrain = cat(2, XTrain{:});
mu = mean(allTrain, 2);
sigma = std(allTrain, 0, 2);
sigma(sigma < 1e-6) = 1;
for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sigma;
end
for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sigma;
end
end

function [XAug, YAug] = augmentTrainingSet(XTrain, YTrain, augmentationCopies, noiseStd, scaleJitter, timeShiftMax, maskFraction, randomSeed)
XAug = XTrain;
YAug = YTrain;
if augmentationCopies <= 0
    return;
end
baseCount = numel(XTrain);
for copyIdx = 1:augmentationCopies
    for i = 1:baseCount
        rng(randomSeed + copyIdx * 100000 + i, "twister");
        seq = XTrain{i};
        scale = 1 + scaleJitter * randn(1, 1, 'single');
        seq = seq * scale;
        seq = seq + noiseStd * randn(size(seq), 'single');
        shift = randi([-timeShiftMax, timeShiftMax], 1, 1);
        if shift ~= 0
            seq = circshift(seq, shift, 2);
        end
        maskWidth = max(1, round(maskFraction * size(seq, 2)));
        maskStart = randi([1, size(seq, 2) - maskWidth + 1], 1, 1);
        seq(:, maskStart:(maskStart + maskWidth - 1)) = 0;
        XAug{end + 1, 1} = seq; %#ok<AGROW>
        YAug(end + 1, 1) = YTrain(i); %#ok<AGROW>
    end
end
end

function options = createTrainingOptions(XVal, YVal, miniBatchSize, maxEpochs, initialLearnRate, validationPatience, validationFrequency)
options = trainingOptions("adam", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=maxEpochs, ...
    InitialLearnRate=initialLearnRate, ...
    Shuffle="every-epoch", ...
    GradientThreshold=1, ...
    GradientThresholdMethod="global-l2norm", ...
    L2Regularization=1e-4, ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=validationFrequency, ...
    ValidationPatience=validationPatience, ...
    OutputNetwork="best-validation", ...
    Verbose=false);
end

function layers = createNetwork(numFeatures, sequenceLength, classNames, embedDim, numHeads, keyDim, dropoutProbability, classWeights)
numClasses = numel(classNames);
layers = layerGraph();

layers = addLayers(layers, sequenceInputLayer(numFeatures, Normalization="none", MinLength=sequenceLength, Name="input"));
layers = addLayers(layers, convolution1dLayer(5, 64, Padding="same", Name="conv1"));
layers = addLayers(layers, batchNormalizationLayer(Name="bn1"));
layers = addLayers(layers, reluLayer(Name="relu1"));
layers = addLayers(layers, convolution1dLayer(3, embedDim, Padding="same", Name="conv2"));
layers = addLayers(layers, batchNormalizationLayer(Name="bn2"));
layers = addLayers(layers, reluLayer(Name="relu2"));
layers = addLayers(layers, positionEmbeddingLayer(embedDim, sequenceLength, Name="pos"));
layers = addLayers(layers, additionLayer(2, Name="add_pos"));
layers = addLayers(layers, layerNormalizationLayer(Name="ln1"));
layers = addLayers(layers, selfAttentionLayer(numHeads, keyDim, ...
    OutputSize=embedDim, DropoutProbability=dropoutProbability, Name="attn1"));
layers = addLayers(layers, additionLayer(2, Name="add_attn1"));
layers = addLayers(layers, layerNormalizationLayer(Name="ln2"));
layers = addLayers(layers, selfAttentionLayer(numHeads, keyDim, ...
    OutputSize=embedDim, DropoutProbability=dropoutProbability, Name="attn2"));
layers = addLayers(layers, additionLayer(2, Name="add_attn2"));
layers = addLayers(layers, layerNormalizationLayer(Name="ln3"));
layers = addLayers(layers, convolution1dLayer(1, 2 * embedDim, Name="ffn1"));
layers = addLayers(layers, reluLayer(Name="relu_ffn"));
layers = addLayers(layers, dropoutLayer(dropoutProbability, Name="drop"));
layers = addLayers(layers, convolution1dLayer(1, embedDim, Name="ffn2"));
layers = addLayers(layers, additionLayer(2, Name="add_ffn"));
layers = addLayers(layers, layerNormalizationLayer(Name="ln4"));
layers = addLayers(layers, globalAveragePooling1dLayer(Name="gap"));
layers = addLayers(layers, fullyConnectedLayer(64, Name="fc1"));
layers = addLayers(layers, reluLayer(Name="relu_fc"));
layers = addLayers(layers, dropoutLayer(dropoutProbability, Name="drop_fc"));
layers = addLayers(layers, fullyConnectedLayer(numClasses, Name="fc_out"));
layers = addLayers(layers, softmaxLayer(Name="softmax"));
layers = addLayers(layers, classificationLayer(Classes=cellstr(classNames), ClassWeights=classWeights, Name="classoutput"));

layers = connectLayers(layers, "input", "conv1");
layers = connectLayers(layers, "conv1", "bn1");
layers = connectLayers(layers, "bn1", "relu1");
layers = connectLayers(layers, "relu1", "conv2");
layers = connectLayers(layers, "conv2", "bn2");
layers = connectLayers(layers, "bn2", "relu2");
layers = connectLayers(layers, "relu2", "pos");
layers = connectLayers(layers, "relu2", "add_pos/in1");
layers = connectLayers(layers, "pos", "add_pos/in2");
layers = connectLayers(layers, "add_pos", "ln1");
layers = connectLayers(layers, "ln1", "attn1");
layers = connectLayers(layers, "attn1", "add_attn1/in1");
layers = connectLayers(layers, "add_pos", "add_attn1/in2");
layers = connectLayers(layers, "add_attn1", "ln2");
layers = connectLayers(layers, "ln2", "attn2");
layers = connectLayers(layers, "attn2", "add_attn2/in1");
layers = connectLayers(layers, "add_attn1", "add_attn2/in2");
layers = connectLayers(layers, "add_attn2", "ln3");
layers = connectLayers(layers, "ln3", "ffn1");
layers = connectLayers(layers, "ffn1", "relu_ffn");
layers = connectLayers(layers, "relu_ffn", "drop");
layers = connectLayers(layers, "drop", "ffn2");
layers = connectLayers(layers, "ffn2", "add_ffn/in1");
layers = connectLayers(layers, "add_attn2", "add_ffn/in2");
layers = connectLayers(layers, "add_ffn", "ln4");
layers = connectLayers(layers, "ln4", "gap");
layers = connectLayers(layers, "gap", "fc1");
layers = connectLayers(layers, "fc1", "relu_fc");
layers = connectLayers(layers, "relu_fc", "drop_fc");
layers = connectLayers(layers, "drop_fc", "fc_out");
layers = connectLayers(layers, "fc_out", "softmax");
layers = connectLayers(layers, "softmax", "classoutput");
end

function label = inferLabel(fileName)
if contains(fileName, "出血")
    label = "出血";
elseif contains(fileName, "排空")
    label = "排空";
elseif contains(fileName, "梗阻")
    label = "梗阻";
elseif contains(fileName, "正常") || contains(fileName, "空-基线")
    label = "正常";
else
    label = "";
end
end

function saveConfusionChart(confusionCounts, classNames, chartTitle, outputPath)
fig = figure('Visible', 'off', 'Color', 'w');
h = heatmap(cellstr(classNames), cellstr(classNames), confusionCounts, 'CellLabelFormat', '%.2f');
h.Title = chartTitle;
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.FontName = 'Helvetica';
h.FontSize = 12;
h.Colormap = parula;
exportgraphics(fig, outputPath, 'Resolution', 200);
close(fig);
end

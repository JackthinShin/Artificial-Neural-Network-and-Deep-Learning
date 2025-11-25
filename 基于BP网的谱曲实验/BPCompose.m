score = [1,3,5,6,5,3,1,3,5,1,7,1, ...
         1,3,5,6,5,3,1,3,5,1,7,1, ...
         6,1,3,5,3,1,6,5,3,5,6,3, ...
         1,3,5,6,5,3,1,7,1,3,7,6,5];
% 舒伯特《小夜曲》
group_size = 5;
num_groups = length(score) - group_size;

P = zeros(group_size, num_groups);
T = zeros(1, num_groups);

for i = 1:num_groups
    P(:, i) = score(i:i+group_size-1)';
    T(i) = score(i+group_size);
end

l = 3:12;
error = zeros(1,10);
bestNodeNum = 1;
for i =1:10
    NodeNum = l(i);%隐含层节点数
    TypeNum = 1;%输出维数
    Epochs = 1000;%训练次数
    TF1 = 'tansig';  TF2 = 'purelin';%设置传递函数
    net = newff(minmax(P),[NodeNum TypeNum],{TF1 TF2},'trainlm');
    net.trainParam.epochs = Epochs;%最大训练次数
    net.trainParam.goal = 1e-8;%最小均方误差
    net.trainParam.min_grad = 1e-8;%最小梯度
    net.trainParam.show = 200;%训练显示间隔
    net.trainParam.time = inf;%最大训练时间
    net = train(net,P,T);%训练

    X = sim(net, P);%测试，输出为预测值
	for j = 1:num_groups
	    error(i) = error(i) + abs(X(j)- T(j))/num_groups;%计算平均绝对误差
    end
    if error(i)<error(bestNodeNum)
        bestNodeNum = i;
    end
end

NodeNum = l(bestNodeNum);%隐含层节点数
TypeNum = 1;%输出维数
Epochs = 1000;%训练次数
TF1 = 'tansig';  TF2 = 'purelin';%设置传递函数
net = newff(minmax(P),[NodeNum TypeNum],{TF1 TF2},'trainlm');
net.trainParam.epochs = Epochs;%最大训练次数
net.trainParam.goal = 1e-8;%最小均方误差
net.trainParam.min_grad = 1e-8;%最小梯度
net.trainParam.show = 200;%训练显示间隔
net.trainParam.time = inf;%最大训练时间
net = train(net,P,T);%训练

predictnum = 40;
predicted_notes = zeros(1, predictnum);  % 存储预测的10个音符
current_sequence = score(end-group_size+1:end)';  % 从最后5个音符开始预测

for i = 1:predictnum
    % 预测下一个音符
    next_note = sim(net, current_sequence);
    predicted_notes(i) = max(1, min(14, round(next_note)));
    
    % 更新序列：移除第一个元素，加入新预测的音符
    current_sequence = [current_sequence(2:end); predicted_notes(i)];
    
    fprintf('预测的第%d个音符: %.4f\n', i, predicted_notes(i));

    score = [score, predicted_notes(i)];

    for j = i:num_groups+i-1
        P(:, j) = score(j:j+group_size-1)';
        T(j) = score(j+group_size);
    end
    net = train(net,P,T);%训练

end

complete_music = [score, predicted_notes];

simple_piano_player(complete_music, 0.3);

function simple_piano_player(melody, tempo)
    % melody: 数字简谱数组 [1,2,3,4,5,6,7]
    % tempo: 速度 (默认0.3)
    
    if nargin < 2
        tempo = 0.3; % 默认速度
    end
    
    % C大调音阶频率 (C4到B4)
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, ...
             523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77];    
    fs = 44100; % 采样率
    music = [];
    
    for i = 1:length(melody)
        note_num = melody(i);
        t = 0:1/fs:tempo;
        freq = freqs(note_num);
        
        % 钢琴音色模拟 (基波 + 谐波)
        wave = sin(2*pi*freq*t) + ...
               0.5*sin(2*pi*2*freq*t) + ...
               0.25*sin(2*pi*3*freq*t);
        % 电子音色
%         wave = sawtooth(2*pi*freq*t, 0.5) + 0.3*sawtooth(2*pi*2*freq*t, 0.5);
        % 吉他音色
%         wave = sin(2*pi*freq*t) .* exp(-0.5*t) + 0.2*sin(2*pi*2*freq*t) .* exp(-0.7*t);
        % 弦乐音色
%         wave = sin(2*pi*freq*t) + 0.4*sin(2*pi*1.5*freq*t) + 0.2*sin(2*pi*2.5*freq*t);

        % 包络处理
        attack = linspace(0, 1, length(t)*0.1);
        release = linspace(1, 0, length(t)*0.2);
        sustain = ones(1, length(t) - length(attack) - length(release));
        envelope = [attack, sustain, release];
        
        wave = wave .* envelope;
        music = [music, wave];
    end
    
    music = music / max(abs(music));
    sound(music, fs);
    
    audiowrite('piano_melody.wav', music, fs);
    fprintf('已保存为: piano_melody.wav\n');
end
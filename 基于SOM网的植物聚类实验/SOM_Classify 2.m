% 是否开花 是否常绿 是否木本 是否水生 
% 是否耐旱 高度等级1–10 是否具香气 寿命等级 1–10
D = [
    1  0  0  0  0  3  1  6;   % 玫瑰
    1  1  0  0  1  2  0  8;   % 仙人掌
    0  1  1  0  1  8  0 10;   % 松树
    1  1  0  0  0  6  0  7;   % 竹子
    1  0  0  1  0  5  0  4;   % 荷花
    1  0  0  0  0  1  0  2;   % 小麦
    0  1  0  0  0  2  0  5;   % 蕨类
    1  0  1  0  0  7  0  9;   % 苹果树
    1  0  0  0  0  3  1  6;   % 薰衣草
    0  1  0  0  1  1  0  3    % 苔藓
]';

plants = {'玫瑰', '仙人掌', '松树', '竹子', '荷花', '小麦', '蕨类', '苹果树', '薰衣草', '苔藓'};

P = normalize(D, 'range');

net = newsom(minmax(P), [3 3]);
net.trainParam.epochs = 200;
net = train(net, P);

y = sim(net, P);
class_id = vec2ind(y);

unique_classes = unique(class_id);
for c = unique_classes
    fprintf("类别 %d: ", c);
    idx = find(class_id == c);
    for k = idx
        fprintf("%s ", plants{k});
    end
    fprintf("\n");
end

% load data
net = load('citeseer-undirected.mat');
A = net.network;
group = net.group;

N = size(A,1);

D = sparse(1:N,1:N,1./sqrt(sum(A,2)));
A = D*A*D; % symmetric transition matrix

A = (A + A*A + A*A*A + A*A*A*A)/4;

% symmetric PPMI
D = sparse(1:N,1:N,1./sqrt(sum(A,2)));
A = D*A*D;
network = max(log(A) - log(1/N), 0);
network = sparse(network);
save('citeseer-PPMI-4.mat', 'network', 'group');

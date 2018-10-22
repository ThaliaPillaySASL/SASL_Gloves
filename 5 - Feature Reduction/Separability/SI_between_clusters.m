clear; 

%Importing the information
fileID = fopen('pq.csv', 'r');
format = '%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
delimiter = ',';
data = textscan(fileID, format, 'Delimiter', delimiter);
fclose(fileID);
array = [data{1:end-1}];
X = array(:,2:end); %extracts sensor inputs
%X = standardise(X); %maps to  range 0 to 1
%Xmeans = mean(X);                 % column means of matrix X
%M = repmat(Xmeans,size(X,1),1);   % replicate rows to match 
%s = std(X);                       % columns std devs of X 
%X = (X - M)*diag(1./s);           % scale cols by 1/col std devs  
X = X +10*randn(size(X));
t = array(:,1); %extracts targets
p = length(t);
d2=dist2(X,X);
[S,I] = sort(d2);
t1 = t(I(1,:));
t2 = t(I(2,:));
pl = sum(t1==t2)/p;


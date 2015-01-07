function y = ker_value_sensing1(x1,x2,W,n)
% kernel value of the sensing1 relaxation--
% TODO: normalize counts to frequency -> rescale by n -> take Gamma

% calculate kernal value for document x1, and x2, with vocabulary size W
% x1, x2 are two W-dimensional vector; 
% x1(w) = number of times word w appears in document x1
% n: scaling parameter in sensing1 relaxation;  
% suggest: set it close to average length of all documents

if nargin<4
    n = 150; % just in case
end

% check input
if length(x1)~=W || length(x2)~=W
   display('error in input');
   y=0;
   return
end

x1 = x1(:);
x2 = x2(:);
N1 = sum(x1);
N2 = sum(x2);
% check if empty file
if N1 ==0 || N2 ==0
    display('empty files')
    y = 0;
    return;
end
% normalize files
vec1 = x1/N1;
vec1 = full(vec1);
vec1 = n * vec1;

vec2 = x2/N2;
vec2 = full(vec2);
vec2 = n * vec2;

tempvec = vec1+vec2;
% calculate kernal
y = sum(gammaln(tempvec+1))-sum(gammaln(vec1+1))-sum(gammaln(vec2+1));



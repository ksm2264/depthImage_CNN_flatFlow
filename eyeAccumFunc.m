function [mindex] = eyeAcumFunc(candVals)


dists = candVals(:,1);

indeces = candVals(:,2);

[~,minDistDex] = min(dists);

mindex = indeces(minDistDex);





end
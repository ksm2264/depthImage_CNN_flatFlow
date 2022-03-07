function         objStruct = computeTriNorms(objStruct)







v1 = objStruct.v(objStruct.f(:,1),:);
v2 = objStruct.v(objStruct.f(:,2),:);
v3 = objStruct.v(objStruct.f(:,3),:);


triNorm = normr(cross(normr(v3-v1),normr(v2-1)));
triNorm = triNorm.*sign(triNorm(:,2));


objStruct.fn = triNorm;
end
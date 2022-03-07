function [walkData,pc] = getSubjWalkData(subj_str,walkNum)


matPath = '/media/karl/DATA/pupilShadowMesh/';
meshPath = '/media/karl/DATA/allMeshes/';


pc_path = [meshPath subj_str '_' num2str(walkNum-1) '_out/texturedMesh.ply'];
mat_path = [matPath subj_str '_' num2str(walkNum) '_pupilShadowMesh.mat'];

pc = pcread(pc_path);

pc_colors = pc.Color;
walkData=load(mat_path);

pc = pointCloud(pc.Location*walkData.orig2alignedMat);
pc.Color = pc_colors;





end


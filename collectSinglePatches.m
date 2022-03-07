close all
clearvars

matPath = '/media/karl/DATA/pupilShadowMesh/';
meshPath = '/media/karl/DATA/allMeshes/';
treePath = '/media/karl/DATA/trees2D/';
treePath3D = '/media/karl/DATA/trees3D/';
nodesPath = '/media/karl/DATA/nodes/';

debugPlot = false;
slant_max = max([32 33 30 31 25 22]);

ecc = 25;



%% defining ret polar grid

[xx,yy] = meshgrid(1:100,1:100);

xx = xx - 50.5;
yy = yy - 50.5;
zz = 125.5*ones(size(xx));

retVecs = cat(3,xx,yy,zz);

retVecs = normalize(retVecs,3,'norm');
retVecs = reshape(retVecs,[numel(xx) 3]);

rho = 2*atan2(vecnorm(retVecs-[0,0,1],2,2),vecnorm(retVecs+[0,0,1],2,2));
theta  = atan2(retVecs(:,2),retVecs(:,1));

eyeTree = KDTreeSearcher([theta(:) rho(:)]);

rho_cutoff_mask = reshape(rho,size(xx))<=pi/8;

%%
for subj_idx =3:10
    if subj_idx<=2
        switch subj_idx
            case 1
                subj_str = 'JAC';
            case 2
                subj_str = 'JAW';
        end
    else
        subj_str = ['s' num2str(subj_idx)];
    end
    
    %%
    load(['/media/karl/DATA/allWalks/' subj_str '.mat']);
    try
        walkTypes = cellfun(@(x) x.walkType,allWalks,'UniformOutput',false);
    catch
        walkTypes = cellfun(@(x) x.trialType,allWalks,'UniformOutput',false);
    end
    
    for walk_num = find(strcmp(walkTypes,'rocks')|strcmp(walkTypes,'Rocks'))
        %     for walk_num = 4:6
        %         try
        %             if exist(['straight_hod_tortuosity_rel/' subj_str '_' num2str(walk_num) '.mat'])~=2
        [walkStruct,pc] = getSubjWalkData(subj_str,walk_num);
        walkStruct.subj = subj_str;
        walkStruct.walk_num = walk_num;
        
        %         load([meshPath subj_str '_' num2str(walk_num-1) '_out/triMesh.mat']);
        %
        %         objStruct.v = objStruct.v*walkStruct.orig2alignedMat;
        %         objStruct.vn = objStruct.vn*walkStruct.orig2alignedMat;
        %
        %         objStruct.vn = objStruct.vn.*sign(objStruct.vn(:,2));
        %         pc.Normal = objStruct.vn(knnsearch(objStruct.v,pc.Location),:);
        %             end
        %
        %         if exist([treePath subj_str '_' num2str(walk_num) '.mat'],'file')~=2
        %             kd_tree_xz = KDTreeSearcher(pc.Location(:,[1 3]));
        %             save([treePath subj_str '_' num2str(walk_num) '.mat'],'kd_tree_xz');
        %         else
        %             load([treePath subj_str '_' num2str(walk_num) '.mat']);
        %         end
        %
        %         %         footLocsAll = pc.Location(knnsearch(kd_tree_xz,footLocsAll(:,[1 3])),:);
        %
        %         %%
        %         if debugPlot
        %             clear triCens
        %             % iterate over dimensions and calculate triangle center coordinate
        %             for dim = 1:3
        %                 this_dim_vals = objStruct.v(objStruct.f',dim);
        %                 this_dim_vals = reshape(this_dim_vals,[3 size(objStruct.f,1)]);
        %                 triCens(:,dim) = mean(this_dim_vals,1)';
        %             end
        %
        %             triCen_tree = KDTreeSearcher(triCens);
        %
        %             % assign colors to triangles based on nearest neighbor
        %             triColors = pc.Color(knnsearch(pc.Location,triCens),:);
        %
        %
        %             objStruct = computeTriNorms(objStruct);
        %             %
        %         end
        %%
        
        stepLength = median(vecnorm(diff(walkStruct.step_plantfoot_xyz(:,3:5)),2,2));
        legLength = stepLength/0.4*0.45;
        
        stepVar = std(vecnorm(diff(walkStruct.step_plantfoot_xyz(:,3:5)),2,2));
        footLength = stepLength/0.4/6.6;
        footFrames = walkStruct.step_plantfoot_xyz(:,1);
        footLocs = walkStruct.step_plantfoot_xyz(:,3:5);
        cens = walkStruct.cens;
        
        %         if exist([nodesPath subj_str '_' num2str(walk_num) '.mat'])~=2
        %             pc_ds = pcdownsample(pc,'gridAverage',footLength/8);
        %             slant_field = rad2deg(acos(pc.Normal(:,2)));
        %             us_pointer = knnsearch(pc_ds.Location,pc.Location);
        %             slant_ds = accumarray(us_pointer,slant_field,[],@median);
        %             xyz_ds = pc_ds.Location;
        %             G = scatteredInterpolant(double(xyz_ds(:,1)),double(xyz_ds(:,2)),double(xyz_ds(:,3)),double(slant_ds));
        %             xyz = double(pc.Location);
        %             slant_field_smoothed = G(xyz(:,1),xyz(:,2),xyz(:,3));
        %             slant_cutoff = slant_field_smoothed<slant_max;
        %
        %             save([nodesPath subj_str '_' num2str(walk_num) '.mat'],'slant_cutoff');
        %         else
        %             load([nodesPath subj_str '_' num2str(walk_num) '.mat']);
        %         end
        %
        %
        %         pc_steppable = select(pc,slant_cutoff);
        %         closestFootLocs = pc_steppable.Location(knnsearch(pc_steppable.Location,footLocs),:);
        %
        eyeVec = walkStruct.eyeVec;
        
        step = walkStruct.step_plantfoot_xyz;
        
        %%
        
        if exist(['/media/karl/DATA/retinalCNN_data/' subj_str '_' num2str(walk_num) ],'dir')~=2
            
            unix(['mkdir ' ['/media/karl/DATA/retinalCNN_data/' subj_str '_' num2str(walk_num) ] ]);
            
        end
%         parfor fr_idx = 1:length(cens)
        for fr_idx = 1:length(cens)
            try
                gaze = eyeVec(fr_idx,:);
                head = cens(fr_idx,:);
                
                vertAng = 2*atan2(norm(gaze-[0 -1 0]),norm(gaze+[0 -1 0]));
                
                
                
                %             pointDirs = normr(pc.Location-head);
                %
                %             pointAngs = 2*atan2(vecnorm(gaze-pointDirs,2,2),vecnorm(gaze+pointDirs,2,2));
                %
                %             usePoints = pointAngs<deg2rad(ecc);
                %
                
                eyeRight = normr(cross(gaze,[0 1 0]));
                eyeUp = normr(cross(eyeRight,gaze));
                eyeRotm = [eyeRight;-eyeUp;gaze];
                
                
                %             thesePoints = pc.Location(usePoints,:);
                %
                %             thesePoints_headRel = thesePoints-head;
                %
                %             thesePoints_dirs = pointDirs(usePoints,:);
                %
                %             thesePoints_dirs_eyeRef = thesePoints_dirs*eyeRotm';
                %
                %             points_theta = atan2(thesePoints_dirs_eyeRef(:,2),thesePoints_dirs_eyeRef(:,1));
                %             points_rho = 2*atan2(vecnorm(thesePoints_dirs_eyeRef-[0 0 1],2,2),...
                %                 vecnorm(thesePoints_dirs_eyeRef+[0 0 1],2,2));
                %
                %             ret_idx = knnsearch(eyeTree,[points_theta(:) points_rho(:)]);
                %
                %             ret_dist_cand = vecnorm(thesePoints_headRel,2,2);
                %
                %             ret_dist = accumarray(ret_idx,ret_dist_cand,[],@min);
                %
                %             if length(ret_dist)<numel(xx)
                %                ret_dist(end:numel(xx))=0;
                %             end
                %             ret_dist = reshape(ret_dist,size(xx));
                %
                %             pull_dex = knnsearch(ret_dist_cand,ret_dist(:));
                %
                %             cand_colors = pc.Color(usePoints,:);
                %
                %             r_channel = cand_colors(pull_dex,1);
                %             g_channel = cand_colors(pull_dex,2);
                %             b_channel = cand_colors(pull_dex,3);
                %
                %             rgb = cat(3,reshape(r_channel,size(xx)),...
                %                 reshape(g_channel,size(xx)),...
                %                 reshape(b_channel,size(xx)));
                %
                %             solid_block = imfill(ret_dist~=0,'holes');
                %
                %             fill_these = solid_block&ret_dist==0;
                %             ret_dist(fill_these)=nan;
                %             ret_dist = fillmissing(ret_dist,'nearest');
                %
                %             for dim = 1:3
                %                 this_chan = double(rgb(:,:,dim));
                %                 this_chan(fill_these)=nan;
                %                 this_chan = fillmissing(this_chan,'nearest');
                %                 rgb(:,:,dim) = uint8(this_chan);
                %             end
                
                next_5_step_idx = find(step(:,1)>=fr_idx,5,'first');
                
                stepLocs = step(next_5_step_idx,3:5);
                
                stepDirs_rel = normr(stepLocs-head)*eyeRotm';
                
                stepDirs_rho = 2*atan2(vecnorm(stepDirs_rel-[0 0 1],2,2),vecnorm(stepDirs_rel+[0 0 1],2,2));
                stepDirs_theta = atan2(stepDirs_rel(:,2),stepDirs_rel(:,1));
                
                stepDirs_img_idx = knnsearch(eyeTree,[stepDirs_theta stepDirs_rho]);
                
                useSteps = stepDirs_rho<=pi/8;
                
                [step_ii,step_jj] = ind2sub(size(xx),stepDirs_img_idx(useSteps));
                
                %             rgb = imread(['/media/karl/DATA/retinalImageRGB/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.png']);
                depth = load(['/media/karl/DATA/retinalImageDepth/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.mat']).map;
                
                rgb = zeros(size(depth,1),size(depth,2),3);
                
                rgb = rgb(76:(250-75),76:(250-75),:);
                depth = depth(76:(250-75),76:(250-75));
                
                depth(depth==65504)=nan;
                rgb = im2uint8(rgb);
                %
                for dim = 1:3
                    
                    this_chan = rgb(:,:,dim);
                    this_chan(~rho_cutoff_mask)=70;
                    %
                    rgb(:,:,dim)=this_chan;
                end
                depth(~rho_cutoff_mask)=nan;
                
                this_foot_map = zeros(size(xx));
                
                this_foot_map(stepDirs_img_idx(useSteps))=1;
                this_foot_map = imgaussfilt(this_foot_map,5);
                
                %             figure(1)
                %             hold off
                %             imshow(rgb)
                %             hold on
                %             plot(step_jj,step_ii,'r.','markersize',24);
                %             hold on
                %             gg = imagesc(this_foot_map);
                %             set(gg,'alphadata',0.5);
                %             drawnow
                
                
                par_save_func(['/media/karl/DATA/retinalCNN_data/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.mat'],...
                    this_foot_map,rgb,depth,step_ii,step_jj);
            catch
            end
            disp([subj_str '_' num2str(walk_num) ': ' num2str(fr_idx/length(cens))]);
        end
        
        
    end
end


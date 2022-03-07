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

myRetVecs = retVecs;

%%
for subj_idx =1:2
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
      
        
        walkStruct = load([matPath subj_str '_' num2str(walk_num) '_pupilShadowMesh.mat']);

        eyeVecs = walkStruct.eyeVec;
        cens = walkStruct.cens;
        
        step = walkStruct.step_plantfoot_xyz;
         for idx = step(1,1):length(cens)
            
            this_plantfoot_xyz(idx,:) = step(find(step(:,1)<=idx,1,'last'),3:5);
            
        end
        if step(1,1)>1
            for ee = 1:step(1,1)-1
                this_plantfoot_xyz(ee,:) = this_plantfoot_xyz(step(1,1),:);
            end
        end
        %%
        for fr_idx = 1:length(cens)-1
            
            gaze = eyeVecs(fr_idx,:);
            head = cens(fr_idx,:);
            
            vertAng = 2*atan2(norm(gaze-[0 -1 0]),norm(gaze+[0 -1 0]));
     
            
            eyeRight = normr(cross(gaze,[0 1 0]));
            eyeUp = normr(cross(eyeRight,gaze));
            eyeRotm = [eyeRight;-eyeUp;gaze];
            
            
            next_5_step_idx = find(step(:,1)>=fr_idx,5,'first');
            
            stepLocs = step(next_5_step_idx,3:5);
            
            stepDirs_rel = normr(stepLocs-head)*eyeRotm';
            
            stepDirs_rho = 2*atan2(vecnorm(stepDirs_rel-[0 0 1],2,2),vecnorm(stepDirs_rel+[0 0 1],2,2));
            stepDirs_theta = atan2(stepDirs_rel(:,2),stepDirs_rel(:,1));
            
            stepDirs_img_idx = knnsearch(eyeTree,[stepDirs_theta stepDirs_rho]);
            
            useSteps = stepDirs_rho<=pi/8;
            
            [step_ii,step_jj] = ind2sub(size(xx),stepDirs_img_idx(useSteps));
            
            rgb = imread(['/media/karl/DATA/retinalImageRGB/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.png']);
            depth = load(['/media/karl/DATA/retinalImageDepth/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.mat']).map;
            
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
            
            this_cen = cens(fr_idx,:);
            eyeVec = walkStruct.eyeVec(fr_idx,:);
            intersectXYZ = walkStruct.gazeXYZ(fr_idx,:);
            
            gY = this_plantfoot_xyz(fr_idx,2);
            
            eye_right = normr(cross(eyeVec,[0 1 0]));
            eye_up = normr(cross(eyeVec,eye_right));
            
            eye_2_world = [eye_right;eye_up;eyeVec];
            
            retVecsWorld = myRetVecs*eye_2_world;
            
            eyeHeight = this_cen(2)-gY;
            
            retDepthFlat = -eyeHeight./retVecsWorld(:,2);
            
            retDepthFlat = reshape(retDepthFlat,size(xx));
            
            depth_insert = depth;
            depth_insert(isnan(depth)) = retDepthFlat(isnan(depth));
            
            head_trans_vec = cens(fr_idx+1,:)-cens(fr_idx,:);
            
            eye_next_vec = normr(intersectXYZ-head_trans_vec-this_cen);
            eye_next_right = normr(cross(eye_next_vec,[0 1 0]));
            eye_next_up = normr(cross(eye_next_vec,eye_next_right));
            
            
            eye_next_2_world = [eye_next_right;eye_next_up;eye_next_vec];
            
            gPoints = depth_insert(:).*retVecsWorld;
            gPoints_next = gPoints - head_trans_vec;
            
            retVecs_shift = normr(gPoints_next)*eye_next_2_world';
            
            mag = 2*atan2(vecnorm(retVecs_shift-myRetVecs,2,2),vecnorm(retVecs_shift+myRetVecs,2,2));
            mag = reshape(mag,size(xx));
            
            
            vec1 = myRetVecs;
            vec2 = retVecs_shift;
            
            vec1_rho = 2*atan2(vecnorm(vec1-[0 0 1],2,2),vecnorm(vec1+[0 0 1],2,2));
            vec2_rho = 2*atan2(vecnorm(vec2-[0 0 1],2,2),vecnorm(vec2+[0 0 1],2,2));
            
            vec1_theta = atan2(vec1(:,2),vec1(:,1));
            vec2_theta = atan2(vec2(:,2),vec2(:,1));
            
            dx = cos(vec2_theta).*vec2_rho - cos(vec1_theta).*vec1_rho;
            dy = sin(vec2_theta).*vec2_rho - sin(vec1_theta).*vec1_rho;
            
            ret_dir = atan2(dy,dx);
            ret_dir = reshape(ret_dir,size(xx));
            
            
            
            ret_flow = opticalFlow(mag.*cos(ret_dir),mag.*sin(ret_dir));
            vx = ret_flow.Vx;
            vy = ret_flow.Vy;
            
            
            gPoints_flat = retDepthFlat(:).*retVecsWorld;
            gPoints_flat_next = gPoints_flat - head_trans_vec;
            retVecs_shift_flat = normr(gPoints_flat_next)*eye_next_2_world';
            
            mag_flat = 2*atan2(vecnorm(retVecs_shift_flat-myRetVecs,2,2),vecnorm(retVecs_shift_flat+myRetVecs,2,2));
            mag_flat = reshape(mag_flat,size(xx));
            
            mag_diff = abs(mag_flat-mag);
            
            save(['/media/karl/DATA/retinalCNN_data/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.mat'],...
                'this_foot_map','rgb','depth','step_ii','step_jj','vx','vy','mag_diff');
            disp([subj_str '_' num2str(walk_num) ': ' num2str(fr_idx/length(cens))]);
        end
        
        
    end
end


close all
clearvars

matPath = '/media/karl/DATA/pupilShadowMesh/';
load('v5_masks.mat');

mag_diff_bins = linspace(0,30,51);
%
% for aa = 1:size(masks,1)
%     for bb = 1:size(masks,2)
%         for cc = 1:size(masks,3)
%
%             counts(aa,bb,cc).cumul_hist = zeros(1,50);
%             counts(aa,bb,cc).bin_edges = mag_diff_bins;
%         end
%     end
% end

[xx,yy] = meshgrid(1:250,1:250);


xx = xx - 125.5;
yy = yy - 125.5;

zz = 125.5*ones(size(xx));

myRetVecs = normr([xx(:) yy(:) zz(:)]);

rho_orig = 2*atan2(vecnorm(myRetVecs-[0 0 1],2,2),vecnorm(myRetVecs+[0 0 1],2,2));
rho_orig =rad2deg(rho_orig);

%% theta rho mapping
[new_xx,new_yy] = meshgrid(1:250,1:250);

new_xx = new_xx - 125.5;
new_yy = new_yy - 125.5;

theta = atan2(new_yy,new_xx);
rho = sqrt(new_xx.^2+new_yy.^2)/125*45;

new_xyz = [cos(theta(:)).*sind(rho(:)) sin(theta(:)).*sind(rho(:)) cosd(rho(:))];

targ_mapper = knnsearch(myRetVecs,new_xyz);

%%
for subj_idx = [1 2 3 4 5 7 8 9 10]
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
    
    for walk_num = 1:length(allWalks)
        %     for walk_num = 4:6
        %         try
        %             if exist(['straight_hod_tortuosity_rel/' subj_str '_' num2str(walk_num) '.mat'])~=2
        walkStruct = load([matPath subj_str '_' num2str(walk_num) '_pupilShadowMesh.mat']);
        
        
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
        
        fixFrames = find(walkStruct.fixBool);
        fixFrames = fixFrames(1:end-1);
        for fr_idx_idx = 1:length(fixFrames)
            fr_idx = fixFrames(fr_idx_idx);
%             if exist(['/media/karl/DATA/retinalMotionStats/histograms_mesh_actual/' subj_str '_' num2str(walk_num) '_' num2str(fr_idx) '.mat'],...
%                     'file')~=2
                %             if exist(['/media/karl/DATA/retinalMotionStats/histograms_mag_diff/' subj_str '_' num2str(walk_num) '_' num2str(fr_idx) '.mat'],'file')~=2
                try
                    %                 rgb = imread(['/media/karl/DATA/retinalImageRGB/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.png']);
                    depth = load(['/media/karl/DATA/retinalImageDepth/' subj_str '_' num2str(walk_num) '/' num2str(fr_idx-1) '.mat']).map;
                    
                    depth(depth==65504)=nan;
                    %                 rgb = im2uint8(rgb);
                    
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
                    
                    gPoint_gaze_flat = -eyeHeight/eyeVec(2)*eyeVec;
                    
                    eye_next_vec_flat = normr(gPoint_gaze_flat-head_trans_vec);
                    eye_next_right_flat = normr(cross(eye_next_vec_flat,[0 1 0]));
                    eye_next_up_flat = normr(cross(eye_next_vec_flat,eye_next_right_flat));
                    
                    
                    eye_next_2_world_flat = [eye_next_right_flat;eye_next_up_flat;eye_next_vec_flat];
                    
                    
                    gPoints_flat = retDepthFlat(:).*retVecsWorld;
                    gPoints_flat_next = gPoints_flat - head_trans_vec;
                    retVecs_shift_flat = normr(gPoints_flat_next)*eye_next_2_world_flat';
                    
                    mag_flat = 2*atan2(vecnorm(retVecs_shift_flat-myRetVecs,2,2),vecnorm(retVecs_shift_flat+myRetVecs,2,2));
                    mag_flat = reshape(mag_flat,size(xx));
                    
                    mag_diff = abs(mag_flat-mag);
                    
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
                    
                    
                    vec1 = myRetVecs;
                    vec2 = retVecs_shift_flat;
                    
                    vec1_rho = 2*atan2(vecnorm(vec1-[0 0 1],2,2),vecnorm(vec1+[0 0 1],2,2));
                    vec2_rho = 2*atan2(vecnorm(vec2-[0 0 1],2,2),vecnorm(vec2+[0 0 1],2,2));
                    
                    vec1_theta = atan2(vec1(:,2),vec1(:,1));
                    vec2_theta = atan2(vec2(:,2),vec2(:,1));
                    
                    dx = cos(vec2_theta).*vec2_rho - cos(vec1_theta).*vec1_rho;
                    dy = sin(vec2_theta).*vec2_rho - sin(vec1_theta).*vec1_rho;
                    
                    ret_dir_flat = atan2(dy,dx);
                    ret_dir_flat = reshape(ret_dir_flat,size(xx));
                    
                    
                    ret_flow = opticalFlow(mag.*cos(ret_dir),mag.*sin(ret_dir));
                    
                    roi_vals = mag_diff(1:125,:);
                    %
                    %
                    %             colormap jet;
                    % %             cmap = colormap;
                    % %             colormap(flipud(cmap));
                    %             figure(1)
                    %             hold off
                    %             imshow(rgb)
                    %             hold on
                    %             gg = imagesc(mag_diff);
                    %             set(gg,'alphadata',0.3);
                    %             caxis(quantile(roi_vals(:),[0.05 0.95]));
                    % %
                    % %             figure(2)
                    % %             clf
                    % %             imagesc(abs(mag_flat-mag));
                    %             drawnow
                    %
                    disp(fr_idx/fixFrames(end));
                    
                    
                    %                 rho_mapped_diff = mag_diff(targ_mapper);
                    %                 rho_mapped_diff = reshape(rho_mapped_diff,size(xx));
                    %
                    %                 rho_mapped_diff = rad2deg(rho_mapped_diff)*30;
                    %
                    %                 rho_mapped_diff(isnan((depth))) = nan;
                    
                    mapped_actual_speed = rad2deg(mag(targ_mapper))*30;
                    mapped_flat_speed = rad2deg(mag_flat(targ_mapper))*30;
                    mapped_actual_dirs = ret_dir(targ_mapper);
                    mapped_flat_dirs = ret_dir_flat(targ_mapper);
                    
                    mapped_actual_speed(isnan(depth))=nan;
                    mapped_flat_speed(isnan(depth))=nan;
                    mapped_actual_dirs(isnan(depth))=nan;
                    mapped_flat_dirs(isnan(depth))=nan;
                    
                    par_save_func_save_both({'/media/karl/DATA/retinalMotionStats/histograms_mesh',...
                        [subj_str '_' num2str(walk_num) '_' num2str(fr_idx) '.mat']},...
                        mapped_actual_speed,mapped_flat_speed,mapped_actual_dirs,mapped_flat_dirs,masks,eyeVec,head_trans_vec);
                catch
                end
%             end
            %             end
        end
        
        
    end
end


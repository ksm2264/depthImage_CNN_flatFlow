function [x_out] = not_dumb_resample(x_in,N)

if size(x_in,1)==1&&size(x_in,2)==3
    x_out = nan*ones(N,3);
else
    
    if size(x_in,2)==1
        
        sample_points_orig = linspace(1,N,length(x_in));
        
        x_out = interp1(sample_points_orig,x_in,1:N);
        
        
    else
        
        for ii = 1:size(x_in,2)
            sample_points_orig = linspace(1,N,size(x_in,1));
            try
                x_out(:,ii) = interp1(sample_points_orig,x_in(:,ii),1:N);
            catch
                keyboard
            end
        end
        
    end
    
    
    
    
end
end


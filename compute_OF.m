function compute_OF(video_names,param)

% create cache folder
dname=sprintf('%s/%s',param.cachepath,'OFYoutube');  %OF; OFBrox
if ~exist(dname,'dir')
    mkdir(dname);
end

fprintf('\n------ Compute OF ------\n')

nb_vid = length(video_names);
impath = param.impath;
cachepath = param.cachepath;
% imext = param.imext;
for vi = 1:nb_vid       % parfor/for
    % get vedio sequences list in the current video
    vidname = video_names{vi};
    
    subimpath = sprintf('%s/%s',impath,vidname);
    subvideo_names = dir(subimpath);
    subvideo_names = {subvideo_names.name};
    subvideo_names = subvideo_names(~ismember(subvideo_names,{'.','..'}));
    
    subnb_vid = length(subvideo_names);
    
    for svi = 1:subnb_vid
        % get image list in the current video
        subvidname = subvideo_names{svi};
        
        [fileNames, numImages] = gzget_training_images(subimpath, subvidname);

        savedir = sprintf('%s/OFYoutube/%s/%s/%s',cachepath,'motorbike',vidname,subvidname); %OF; OFBrox
        if ~exist(savedir,'dir')
            mkdir(savedir);
        end

        fprintf('compute OF: %d out of %d videos\n',vi,nb_vid)

    %     % read test frame
    %     testImage = imread(fileNames{1});   % double(imread(fileNames{l}));
    %     if size(testImage, 3) > 1
    %         testImage = rgb2gray(testImage);
    %     end
    %     [h, w] = size(testImage);

        for i = 1:numImages-1
            % fprintf('compute OF: %d out of %d frames\n',i,length(images)-1)

            [~,imname,~] = fileparts(fileNames{i});
            imsave = sprintf('%s/%s.jpg',savedir,imname);
%             % 1) flowu
%             flowsave = sprintf('%s/%s',savedir,'flowu');
%             flowsave = sprintf('%s%d.mat',flowsave,i);
            % 2) flowuv
            flowsave = sprintf('%s/%s',savedir,'flowuv');
            flowsave = sprintf('%s%d.mat',flowsave,i);
            % Pass the flow which has already been computed
            if exist(flowsave,'file')    % flowsave; imsave
                continue; 
            end
            
            im1 = double(imread(fileNames{i}));   % double(imread(fileNames{fileIndex}));
            im2 = double(imread(fileNames{i+1})); % double(imread(fileNames{fileIndex+1}));
            if size(im1,3)<3
                im1 = cat(3,im1,im1,im1);
            end
            if size(im2,3)<3
                im2 = cat(3,im2,im2,im2);
            end

%             % Optical flow Algorithm 1 -- NLP 
%             uv = estimate_flow_interface(im1, im2,'classic+nlp');  % 'classic+nl-fastp';'classic+nlp';'classic+nl-fast';'classic+nl';

%             % Optical flow Algorithm2 -- THOMAS BROX 2004             
%             uv = mex_OF(im1,im2);
            
            % Optical flow Algorithm3 -- THOMAS BROX 2011
            uv = mex_LDOF(im1, im2);                    
            u = uv(:,:,1);
            v = uv(:,:,2);
            
%             % Some frames are duplicate, ignore flow for those as it will be confusing for the CNN
%             if range(u(:))<0.5 && range(v(:))<0.5 
%                 continue;
%             end

            norm = sqrt(u.^2 + v.^2);
            max_flow = 8;       % maximum absolute value of flow
            scalef = 128/max_flow;
            uv = uv*scalef;     % scale flow
            uv = uv+128;        % center it around 128
            uv(uv<0) = 0;
            uv(uv>255) = 255;   % crop the values below 0 and above 255

            norm = norm*scalef; % same for magnitude
            norm = norm+128;
            norm(norm<0) = 0;
            norm(norm>255) = 255;

            im = uint8(cat(3,uv,norm)); % concatenate flow_x, flow_y and magnitude
            imwrite(im, imsave);
            
            % Optical flow Algorithm3 -- THOMAS BROX 2011
            uv = int16(uv);
            
%             UName = sprintf('flowu%d.mat',i);
%             UNames = fullfile(savedir, UName); 
%             save(UNames,'u');
%             VName = sprintf('flowv%d.mat',i);
%             VNames = fullfile(savedir, VName); 
%             save(VNames,'v');
                     
            UVName = sprintf('flowuv%d.mat',i);
            UVNames = fullfile(savedir, UVName); 
            save(UVNames,'uv');
        end
    end
end


%         % Backward Flow
%         uvback = estimate_flow_interface(im2,im1, 'classic+nl-fastp');
%         InfoNameback  = sprintf('BackwardFlow%03d.flo',i);
%         InfoNamesback = fullfile(savedir, InfoNameback); 
%         writeFlowFile(uvback, InfoNamesback);

            
%             InfoName  = sprintf('ForwardFlow%03d.flo',i);
%             InfoNames = fullfile(savedir, InfoName); 
%             writeFlowFile(uv, InfoNames);
            
%             NormName = sprintf('norm%d.mat',i);
%             NormNames = fullfile(savedir, NormName); 
%             save(NormNames,'norm');


%             img = uint8(flowToColor(uv));
%             outputFileName = sprintf('ColFlow%d.jpg',i);
%             outputFileNames = fullfile(savedir, outputFileName); 
%             imwrite(img, outputFileNames);
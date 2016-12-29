function extract_cnn_OFpatches(video_names,param)

% create cache folders
% cdirs={'patches_app','patches_flow','patches_app/image_patch1','patches_flow/image_patch1','patches_app/left_hand','patches_flow/left_hand', ...
%     'patches_app/right_hand','patches_flow/right_hand','patches_app/upper_body','patches_flow/upper_body', ...
%     'patches_app/full_body','patches_flow/full_body','patches_app/full_image','patches_flow/full_image'};
cdirs={'patches_app','patches_flow','patches_app/image_patch1','patches_flow/image_patch1','patches_app/image_patch2','patches_flow/image_patch2', ...
    'patches_app/full_image','patches_flow/full_image'};

for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir')
        mkdir(dname); 
    end
end

addpath(genpath('inexact_alm_rpcaModify'));

fprintf('\n------ Compute CNN patches based on Motion and Image Salient ------\n')

% methodchoose =1; % original RPCA code
methodchoose = 21; % block RPCA method
SeqsPatches = cell(length(video_names),1);

for vi = 1:length(video_names)        % parfor/for
    fprintf('extract patches .. : %d out of %d videos\n',vi,length(video_names))
    
    % get image list in the current video
    vidname = video_names{vi} ;
    
    % Subsequences Directory
    for d=3:length(cdirs)
        dname=sprintf('%s/%s',param.cachepath,cdirs{d});
        subdname=sprintf('%s/%s',dname,vidname);
        if ~exist(subdname,'dir')
            mkdir(subdname);
        end
    end
    
    subimpath = sprintf('%s/%s',param.impath,vidname);
    subvideo_names = dir(subimpath);
    subvideo_names = {subvideo_names.name};
    subvideo_names = subvideo_names(~ismember(subvideo_names,{'.','..'}));
    
    subnb_vid = length(subvideo_names);
    
    for svi = 1:subnb_vid
        % get image list in the current video
        subvidname = subvideo_names{svi};
        [fileNames, numImages] = gzget_training_images(subimpath, subvidname);
            
        % get video joint positions and human scales
        subposepath = sprintf('%s/%s',param.jointpath,vidname);
        Jointpositions = sprintf('%s/%s/joint_positions.mat',subposepath,subvidname);
        if ~exist(Jointpositions,'file')
            continue;
        end                
        positions=load(Jointpositions);
        scale=positions.scale;
        positions=positions.pos_img;

%         if exist(sprintf('%s/patches_flow/image_patch1/%s/%s_im%05d.jpg',param.cachepath,vidname,subvidname,1),'file')
%             continue;
%         end
        
        suf = {'app','flow'};
        imdirs = {param.impath,sprintf('%s/OF',param.cachepath)};   % OF

    %     imdirpath = imdirs{1};
        flowdirpath = imdirs{2};

        netApp = param.(sprintf('net_%s',suf{1}));
        netFlow = param.(sprintf('net_%s',suf{2}));

        %% read first image to get the size info
    %     impath = sprintf('%s/%s/%s',imdirpath,vidname,images{1});
        testImage = imread(fileNames{1});
        if size(testImage, 3) > 1
            testImage = rgb2gray(testImage);
        end
        [h, w] = size(testImage);

        %% read image file, to establish the data matrix
        data = zeros([h*w, numImages], 'double');
        for fileIndex = 1:numImages
    %         impath = sprintf('%s/%s/%s',imdirpath,vidname,images{fileIndex});
            currentImage = imread(fileNames{fileIndex});
            if size(currentImage, 3) > 1
                currentImage = rgb2gray(currentImage);
            end
            imgs = reshape(currentImage,w*h,1);
            data(:, fileIndex) = imgs(:,1);
        end
        data=data/255.0;

        %%% read flow data, u v and norm %%%
        uflowdata = zeros([h*w, numImages-1], 'double');
        vflowdata = zeros([h*w, numImages-1], 'double');
        magflowdata = zeros([h*w, numImages-1], 'double');
        
        flowDir= fullfile(flowdirpath, vidname);
        flowDir= fullfile(flowDir, subvidname);
        
        for j = 1:numImages - 1
            infName0 = sprintf('flowu%d.mat',j);
            infName00 = fullfile(flowDir, infName0);
            load(infName00,'u');
            utemp = reshape(u,w*h,1);
            uflowdata(:, j) = utemp(:,1);

            infName1 = sprintf('flowv%d.mat',j);
            infName11 = fullfile(flowDir, infName1);
            load(infName11,'v');
            vtemp = reshape(v,w*h,1);
            vflowdata(:, j) = vtemp(:,1);

            infName2 = sprintf('norm%d.mat',j);
            infName22 = fullfile(flowDir, infName2);
            load(infName22,'norm');
            magtemp = reshape(norm,w*h,1);
            magflowdata(:, j) = magtemp(:,1);
        end

        % Compute Pose-based patches
        for i = 1:2 % appearance and flow
            imdirpath = imdirs{i};       
            net=param.(sprintf('net_%s',suf{i}));

            for idim = 1:min(numImages,length(positions))     % min(numImages,length(positions))
                
%                 if exist(sprintf('%s/patches_%s/full_image/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim),'file')
%                     continue;
%                 end

                % Read images (frames and flows)
                if i==1 % appearance
                    im = imread(fileNames{idim});   % impath = sprintf('%s/%s/%s.jpg',imdirpath,vidname,iname);
                else % flow
                    [~,iname,~]=fileparts(fileNames{idim});
                    impath = sprintf('%s/%s/%s/%s.jpg',imdirpath,vidname,subvidname,iname); % flow has been previously saved in JPG
                    if ~exist(impath,'file')
                        continue; 
                    end
                    
%                     uflow = uflowdata(:,idim);
%                     vflow = vflowdata(:,idim);
%                     if range(uflow(:))<0.5 && range(vflow(:))<0.5 
%                         continue;
%                     end
                    
                    im = imread(impath);
                end

%                 % part CNN (fill missing part before resizing)
%                 sc=scale(idim); lside=param.lside*sc;
%                 % left hand
%                 lhand = get_box_and_fill(positions(:,param.lhandposition,idim)-lside,positions(:,param.lhandposition,idim)+lside,im);
%                 lhand = imresize(lhand, net.normalization.imageSize(1:2), net.normalization.interpolation);
% 
%                 % right right
%                 rhand = get_box_and_fill(positions(:,param.rhandposition,idim)-lside,positions(:,param.rhandposition,idim)+lside,im);
%                 rhand = imresize(rhand, net.normalization.imageSize(1:2), net.normalization.interpolation);
% 
%                 % upper body
%                 sc=scale(idim); lside=3/4*param.lside*sc;
%                 upbody = get_box_and_fill(min(positions(:,param.upbodypositions,idim),[],2)-lside,max(positions(:,param.upbodypositions,idim),[],2)+lside,im);
%                 upbody = imresize(upbody, net.normalization.imageSize(1:2), net.normalization.interpolation);
% 
%                 % full body
%                 fullbody = get_box_and_fill(min(positions(:,:,idim),[],2)-lside,max(positions(:,:,idim),[],2)+lside,im);
%                 fullbody = imresize(fullbody, net.normalization.imageSize(1:2), net.normalization.interpolation);

                % full image CNNf (just resize frame)
                fullim = imresize(im, net.normalization.imageSize(1:2), net.normalization.interpolation);

%                 imwrite(lhand,sprintf('%s/patches_%s/left_hand/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
%                 imwrite(rhand,sprintf('%s/patches_%s/right_hand/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
%                 imwrite(upbody,sprintf('%s/patches_%s/upper_body/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
%                 imwrite(fullbody,sprintf('%s/patches_%s/full_body/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
                imwrite(fullim,sprintf('%s/patches_%s/full_image/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
            end
        end

        % Compute Motion Saliency-based patches
        if methodchoose==21             
                         
            % Test save temporal data and load it
            [A_hat, E_hat, iter] = Block2and1norm_inexact_alm_rpca(data);

            % this is for carbluelight case
            [MaskBlock1Pixle2, E_blockmask, PixelMask] = gzEmatrixFindMasksChangingblocksize(E_hat, 0.1, 4);
            PixelEasyMask = gzEasyPixelMask(E_hat, 0.11);

    %         PixelMaskRelativeChange = gzComputeRelativeChangePixelmask(data, A_hat, 0.1);
    %         [MaskBlock1Pixle2 E_blockmask] = gzEmatrixFindMasksChangingblocksize(RelativeChange, 0.2, 4);

            % Motion Salient -- Optical Flow
            PixelSeedMask = gzMotionConsistentAnalysis(MaskBlock1Pixle2, E_blockmask, PixelMask, uflowdata, vflowdata, magflowdata, data);

            TargetCoord = gzGetTarget(PixelEasyMask,PixelSeedMask,uflowdata,vflowdata,magflowdata);

            % Save the Patches
            SeqsPatches{vi} = TargetCoord; 
    %         NormName = sprintf('PatchInf%s.mat',vidname);
    %         save(NormName,'TargetCoord');

            TargetPatchImg0(data,TargetCoord,param,vidname,subvidname,netApp,netFlow,length(positions),uflowdata,vflowdata,magflowdata);

    %            clear A_hat E_hat MaskBlock1Pixle2 E_blockmask PixelMask PixelEasyMask PixelSeedMask;
    %            [A_hat, E_hat, iter] = gzBlockRPCAwithTargetBlockinfo(data);

    %            [Mask_matrix E_blockmask]=gzAnalyzeEmatrix(E_hat);
    %            [A_hat E_hat iter] = gzAdaptiveThreshblock2and1norm_inexact_alm_rpca(data, E_blockmask);%
        end
        
    end
end






%         if exist(sprintf('%s/patches_flow/image_patch1/%s/%s_im%05d.jpg',param.cachepath,vidname,subvidname,j),'file')
%             continue;
%         end
                
%         for idim=1:numImages
%             if exist(sprintf('%s/full_image/%s_im%05d.jpg',param.cachepath,vidname,idim),'file')
%                 continue;
%             end
%             % get image and flow
%             if i == 1 % appearance
%                 impath = sprintf('%s/%s/%s',imdirpath,vidname,images{idim});
%             else % flow
%                 [~,iname,~]=fileparts(images{idim});
%                 impath = sprintf('%s/%s/%s.jpg',imdirpath,vidname,iname);  % flow has been previously saved in JPG
%                 if ~exist(impath,'file'); continue ; end;  % flow was not computed (see compute_OF.m for info)
%             end
%             im = imread(impath);
%             
%             % part CNN (fill missing part before resizing)
%             sc=scale(idim); lside=param.lside*sc ;
%             % left hand
%             lhand = get_box_and_fill(positions(:,param.lhandposition,idim)-lside,positions(:,param.lhandposition,idim)+lside,im);
%             lhand = imresize(lhand, net.normalization.imageSize(1:2)) ;
%             
%             % right right
%             rhand = get_box_and_fill(positions(:,param.rhandposition,idim)-lside,positions(:,param.rhandposition,idim)+lside,im);
%             rhand = imresize(rhand, net.normalization.imageSize(1:2)) ;
%             
%             % upper body
%             sc=scale(idim); lside=3/4*param.lside*sc ;
%             upbody = get_box_and_fill(min(positions(:,param.upbodypositions,idim),[],2)-lside,max(positions(:,param.upbodypositions,idim),[],2)+lside,im);
%             upbody = imresize(upbody, net.normalization.imageSize(1:2)) ;
%             
%             % full body
%             fullbody = get_box_and_fill(min(positions(:,:,idim),[],2)-lside,max(positions(:,:,idim),[],2)+lside,im);
%             fullbody = imresize(fullbody, net.normalization.imageSize(1:2)) ;
%             
%             % full image CNNf (just resize frame)
%             fullim = imresize(im, net.normalization.imageSize(1:2)) ;
%             
%             imwrite(lhand,sprintf('%s/patches_%s/left_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
%             imwrite(rhand,sprintf('%s/patches_%s/right_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
%             imwrite(upbody,sprintf('%s/patches_%s/upper_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
%             imwrite(fullbody,sprintf('%s/patches_%s/full_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
%             imwrite(fullim,sprintf('%s/patches_%s/full_image/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
%         end


%     fileIndex1 = 1;
%     fileIndex10 = 10;
%     fileIndex30 = 30;
%     savedir1 = sprintf('%s/patches_app/image_patch1/%s_im%05d.jpg',param.cachepath,vidname,fileIndex1);
%     savedir2 = sprintf('%s/patches_app/image_patch1/%s_im%05d.jpg',param.cachepath,vidname,fileIndex10);
%     savedir3 = sprintf('%s/patches_app/image_patch1/%s_im%05d.jpg',param.cachepath,vidname,fileIndex30);
%     % Pass the flow which has already been computed
%     if exist(savedir1,'file') && exist(savedir2,'file') && exist(savedir3,'file')
%         continue; 
%     end

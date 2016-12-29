function extract_cnn_OFpatchfeatures(video_names,param)

% create cache folders
% cdirs={'cnn_features_app','cnn_features_app/frame_features','cnn_features_app/video_features' ...
%     'cnn_features_flow','cnn_features_flow/frame_features','cnn_features_flow/video_features'};
cdirs={'cnn_features_app','cnn_features_app/frame_features','cnn_features_app/video_features3' ...
    'cnn_features_flow','cnn_features_flow/frame_features','cnn_features_flow/video_features3'};

for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir'); 
        mkdir(dname); 
    end
end

fprintf('\n------ Extract CNN features ------\n')

for vi = 1:length(video_names)        % parfor/for
    vidname = video_names{vi};
    
    subimpath = sprintf('%s/%s',param.impath,vidname);
    subvideo_names = dir(subimpath);
    subvideo_names = {subvideo_names.name};
    subvideo_names = subvideo_names(~ismember(subvideo_names,{'.','..'}));
    
    subnb_vid = length(subvideo_names);
    
    suf={'app','flow'};
    for i=1:2 % appearance and flow           
        % get folders
        patchesdir = sprintf('%s/patches_%s',param.cachepath,suf{i});
        framefeaturesdir = sprintf('%s/cnn_features_%s/frame_features',param.cachepath,suf{i});
        videofeaturesdir = sprintf('%s/cnn_features_%s/video_features3',param.cachepath,suf{i});

        subdname=sprintf('%s/%s',videofeaturesdir,vidname);
        if ~exist(subdname,'dir'); 
            mkdir(subdname); 
        end

        % get list of part patches
        [filelist,outlist]=get_patches_list(patchesdir,framefeaturesdir,vidname); %frame size:224*224*3

        % get net
        net=param.(sprintf('net_%s',suf{i}));
        if param.use_gpu
            net = vl_simplenn_move(net, 'gpu');
        end         % move net on GPU if needed
        bsize=param.batchsize;
        nim=length(filelist); %Num_clip_frame(each clip contains frames)*Num_Patch

        % extract CNN features per frame
        for b=1:bsize:nim
            fprintf('%s -- feature extraction: %d\tover %d:\t',suf{i},b,nim);tic;
            im = vl_imreadjpeg(filelist(b:min(b+bsize-1,nim)),'numThreads', param.nbthreads_netinput_loading);
            im = cat(4,im{:}); %im=224*224*3*128(frame size:224*224*3; bsize=128->using 128 frames as a group for computation)
            im = bsxfun(@minus, im, net.normalization.averageImage); %Mean subtraction
            if param.use_gpu; im = gpuArray(im); end
            % RES=VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
            res=vl_simplenn(net,im); %res.x=224*224*3*128*(1+Num_Layer)(i.e.net_app"imagenet-vgg-f":22=1+21(Num_Layer);1 is the initial input data-->im)
            fprintf('extract %.2f s\t',toc);tic;
            % res(end-2).x --> Take features after last ReLU; Layer18: ‘fc7’ (full connected layer?FC2 4096); Layer19: ‘relu7’ (last ReLU); 
            % res(end-1).x --> FC3 (Layer20: ‘fc8’ (full connected layer?FC3))
            % res(end).x --> Softmax Probability (Layer21: ‘Softmax’ (probability))
            save_feats(squeeze(res(end-2).x),outlist(b:min(b+bsize-1,nim)),param);%For each frame at each patch, 4096 features are extracted
            fprintf('save %.2f s\n',toc)
        end
        % group frame features in their corresponding video
        % features(1).x <--- left hand
        % features(2).x <--- rigth hand
        % features(3).x <--- upper body
        % features(4).x <--- full body
        % features(5).x <--- full image
        

        for svi = 1:subnb_vid
            % get image list in the current video
            subvidname = subvideo_names{svi};

            subfname = sprintf('%s/full_image/%s',framefeaturesdir,vidname);
            featurepath = sprintf('%s/%s_im*',subfname,subvidname); % flow has been previously saved in JPG
            td = dir(featurepath);
            if length(td)<1
                continue; 
            end

            group_cnn_features(framefeaturesdir,videofeaturesdir,vidname,subvidname);

        end
    end    
end

function [filelist,outlist]=get_patches_list(indirname,outdirname,vidname)
% bodyparts={'left_hand' 'right_hand' 'upper_body' 'full_body' 'full_image'}; %'image_patch1'  
bodyparts={'image_patch1' 'image_patch2' 'full_image'};

images=dir(sprintf('%s/%s/%s/*jpg',indirname,bodyparts{1},vidname));
images = {images.name};
[~,resnames,~]=cellfun(@(x) fileparts(x),images,'UniformOutput',false);
resnames=strcat(resnames,repmat({'.mat'},1,length(images)));

filelist=cell(1,length(bodyparts)*length(images));
outlist=cell(1,length(bodyparts)*length(images));
for i=1:length(bodyparts)
    indirpath=sprintf('%s/%s/%s/',indirname,bodyparts{i},vidname);
    destdirpath=sprintf('%s/%s/%s/',outdirname,bodyparts{i},vidname);
    if ~exist(destdirpath,'dir')
        mkdir(destdirpath); 
    end
    
    pimages=repmat({indirpath},1,length(images));
    qimages=repmat({destdirpath},1,length(images));
    
    filelist(1,1+(i-1)*length(images):i*length(images)) = strcat(pimages,images);
    outlist(1,1+(i-1)*length(images):i*length(images)) = strcat(qimages,resnames);
end

function save_feats(feats,outlist,param)
assert(length(outlist)==size(feats,2));
if param.use_gpu; feats=gather(feats); end

parfor i=1:length(outlist)
    out=outlist{i};
    features=feats(:,i)';
    parsave(out,features);
end

function parsave(out,features)
save(out,'features');

function group_cnn_features(framefeaturesdir,videofeaturesdir,video_name,subvideo_name)
% features(1).x <--- left hand
% features(2).x <--- rigth hand
% features(3).x <--- upper body
% features(4).x <--- full body
% features(5).x <--- full image

% part sub-directories to check
% subD={'left_hand' 'right_hand' 'upper_body' 'full_body' 'full_image'}; %'image_patch1'
subD={'image_patch1' 'image_patch2' 'full_image'};

features = [] ;
for i=1:length(subD)
    dirpath=sprintf('%s/%s/%s',framefeaturesdir,subD{i},video_name);
    pathname=sprintf('%s/%s_im*',dirpath,subvideo_name);
    td=dir(pathname);
    assert(~isempty(td));
    x=zeros(length(td),4096); %length(td)->Num_frames in each video clip; 4096 are the extracted features of each frame patch
    features(i).name=sprintf('CNNf_%s',subD{i});
    for j=1:length(td)
        samplepath=sprintf('%s/%s',dirpath,td(j).name);
        tmp=load(samplepath) ;
        x(j,:)=tmp.features ;
    end
    features(i).x=x; %For each patch, x=VideoClipNum_frames*4096; features:[x1->patch1; x2->patch2,...]
end
save(sprintf('%s/%s/%s.mat',videofeaturesdir,video_name,subvideo_name),'features');

% %Frame Read
% im1 = imread('G:\Test Sequence Database\Middlebury Database\other-color-allframes\RubberWhale\frame10.png'); 

% Video Read
fileName = 'G:\Action Data\UCLA_Aerial_Event\Calibrated_Video\10.avi';  
obj = VideoReader(fileName);
numFrames = obj.NumberOfFrames;  % Frame Number
for k = 1 : numFrames            % Read each frame
    frame = read(obj, k);
    imshow(frame);               % Show frame
    savedir = 'G:\Action Data\UCLA_Aerial_Event\Calibrated_Video';  %Data\UCF-Sports\images\Golf-Swing-Front\002
    FIndex = k-1;
    imsave = sprintf('%s/%d.jpg',savedir,FIndex);
    imwrite(frame,imsave);       % Frame save
end

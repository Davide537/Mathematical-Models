function [] = brain3d(rate, all_areas, t_idx, out_filename)
% t_idx: timestep index to render (default: last timestep = size(rate,2))
% out_filename: explicit path, e.g. 'Fig/3dBrain/frame_0001.png'

if nargin < 3 || isempty(t_idx)
    t_idx = size(rate,2);  % default: last timestep, same as before
end
if nargin < 4 || isempty(out_filename)
    out_filename = sprintf('Fig/3dBrain/brain3d_%s.png', datestr(now,'mmdd_HHMMSS'));
end

load('data/scalablebrainatlasdata.mat')
fig = figure('visible','off', 'Color','w');
ax = axes('Parent', fig);
set(ax, 'Color', 'w')
hold(ax,'on');

colors = 3*ones(length(labels),1);
facesarea = find(labels==17); colors(facesarea,1) = 35;

converter = [28 25 20 90 36 74 88 71 45 21 63 50 86 72 52 75 83 51 12 16 78 84 93 69 41 27 95 89 10 77];

if all_areas
    ratecolor = 10:39;
    ratecolor([0 1 2 4]+1)                  = 10;
    ratecolor([3 6 8 12 20 21 23]+1)        = 20;
    ratecolor([9 11 18 19 25 28]+1)         = 30;
    ratecolor([10 13 14 17 22 24 26 29]+1)  = 40;
    ratecolor([5 7 15 16 27]+1)             = 50;
else
    % Use the requested timestep instead of always 'end'
    ratecolor = squeeze( mean(rate(:,t_idx,1:30),1) ) - squeeze( mean(rate(:,t_idx,31:60),1) );
end

Nareas = length(ratecolor);
areaCentroids = nan(Nareas, 3);
for i = 1:Nareas
    SBAlabel  = converter(i);
    facesarea = find(labels==SBAlabel);
    vIdx      = unique(faces(facesarea,:));
    areaCentroids(i,:)  = mean(vertices(vIdx,:), 1);
    colors(facesarea,1) = ratecolor(i) + 2.5;
end

braincolor = all_areas * jet(64) + (~all_areas) * flipud(autumn(64));
braincolor(1:6,:) = 0.5;
colormap(ax, braincolor);

patch('Parent', ax, 'Vertices',vertices,'Faces',faces,'FaceVertexCData',colors, ...
      'FaceColor','flat','FaceLighting','gouraud','EdgeColor','none','CDataMapping','scaled');

set(fig,'renderer','opengl', 'Color', 'w');
lighting('gouraud'); material([0.4 0.7 0.1 15]);
daspect([1 1 1]);
axis(ax,'off','image');
camlight(ax,-180,25);
camlight(ax,-120,0);
camlight(ax,-50,-20);
view(ax,-90,0);

set(ax,'CLim',[0.3 50]);  % fixed scale — critical for a consistent GIF

if all_areas
    %% ── Camera direction vector (robust, derived from actual camera) ──────────
    cp  = campos(ax);
    ct  = camtarget(ax);
    camDir = cp - ct;
    camDir = camDir / norm(camDir);   % unit vector pointing FROM scene TO camera

    camPush = 15;  % tune as before
    %% ── Draw labels ──────────────────────────────────────────────────────────

    area_list = {'V1','V2','V4','DP','MT','8m','5','8l','2',...
                'TEO','F1','STPc','7A','46d','10','9/46v','9/46d',...
                'F5','TEpd','PBr','7m','LIP','F2','7B','ProM','STPi',...
                'F7','8b','STPr','24c'}

    for i = 1:Nareas
        if any(isnan(areaCentroids(i,:))), continue; end
        vOut = areaCentroids(i,:) + camDir * camPush;

        % Dark backing circle so text is readable over any brain color
        scatter3(ax, vOut(1), vOut(2), vOut(3), 300, 'w', 'filled', ...
                'MarkerFaceAlpha', 0.6);

        % White label on top
        text(ax, vOut(1), vOut(2), vOut(3), area_list{i}, ...
            'FontSize', 7, ...
            'Color', 'black', ...
            'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment',   'middle', ...
            'Clipping', 'off');
    end
else
    cb = colorbar(ax,'Position',[0.93 0.15 0.03 0.7]);
    cb.Color = 'k';
    ylabel(cb,'Firing rate (Hz)', 'Color','k');
end

drawnow;
exportgraphics(fig, out_filename, 'Resolution', 300, 'BackgroundColor', 'w');
[img, ~, ~] = imread(out_filename);
mask = img(:,:,1) == 255 & img(:,:,2) == 255 & img(:,:,3) == 255;
alpha = uint8(~mask) * 255;
imwrite(img, out_filename, 'Alpha', alpha);
close(fig);
end

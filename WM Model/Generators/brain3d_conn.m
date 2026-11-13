function [] = brain3d_conn(conn, out_filename, opts)
% conn:         30x30 connectivity matrix (e.g. FLN or W_E block), in the SAME
%               order as area_list / converter below (the model's order, not hierarchy order).
%               conn(i,j) = weight from area j to area i.
% out_filename: output png path
% opts (struct, all optional):
%   .keep_frac   (0.35)   fraction of strongest edges to draw
%   .log_scale   (true)   map weights on log10 scale for line width
%   .lw_range    ([0.3 4])
%   .view        ([-90 0])  azimuth/elevation ([-90 0] lateral, [0 90] top)

if nargin < 2 || isempty(out_filename)
    out_filename = sprintf('Fig/3dBrain/brain3d_conn_%s.png', datestr(now,'mmdd_HHMMSS'));
end
if nargin < 3, opts = struct(); end
def = struct('keep_frac',0.35, 'log_scale',true,'lw_range',[0.3 4],'view',[-90 0]);

fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(opts, fn{k}) || isempty(opts.(fn{k})), opts.(fn{k}) = def.(fn{k}); end
end

load('C:\Users\david\OneDrive\Documenti\GitHub\Mathematical Models\WM Model\data\scalablebrainatlasdata.mat')

converter = [28 25 20 90 36 74 88 71 45 21 63 50 86 72 52 75 83 51 12 16 78 84 93 69 41 27 95 89 10 77];
area_list = {'V1','V2','V4','DP','MT','8m','5','8l','2',...
             'TEO','F1','STPc','7A','46d','10','9/46v','9/46d',...
             'F5','TEpd','PBr','7m','LIP','F2','7B','ProM','STPi',...
             'F7','8b','STPr','24c'};
Nareas = numel(converter);

fig = figure('visible','off','Color','w');
ax = axes('Parent',fig); hold(ax,'on');

%% Centroids
areaCentroids = nan(Nareas,3);
for i = 1:Nareas
    facesarea = find(labels==converter(i));
    vIdx = unique(faces(facesarea,:));
    areaCentroids(i,:) = mean(vertices(vIdx,:),1);
end

%% Edge selection (undirected: use the stronger direction of each pair)
C = conn; C(1:Nareas+1:end) = 0;          % remove self-connections
U = max(C, C');                            % symmetric strength
[ii,jj] = find(triu(U,1) > 0);
w = U(sub2ind(size(U), ii, jj));
[w, ord] = sort(w,'ascend'); ii = ii(ord); jj = jj(ord);   % draw weak first
nKeep = max(1, round(opts.keep_frac * numel(w)));
sel = numel(w)-nKeep+1 : numel(w);
ii = ii(sel); jj = jj(sel); w = w(sel);

if opts.log_scale, wv = log10(w); else, wv = w; end
wn = (wv - min(wv)) / max(eps, (max(wv) - min(wv)));       % 0..1
lw = opts.lw_range(1) + wn * diff(opts.lw_range);

for e = 1:numel(w)
    P = areaCentroids([ii(e) jj(e)],:);
    shade = 0.85 - 0.65*wn(e);             % weak = light red, strong = dark red
    plot3(ax, P(:,1), P(:,2), P(:,3), '-', 'LineWidth', lw(e), ...
          'Color', [0.9 shade*0.25 shade*0.25]);
end

%% Nodes + labels
scatter3(ax, areaCentroids(:,1), areaCentroids(:,2), areaCentroids(:,3), ...
         25, 'k', 'filled');
for i = 1:Nareas
    text(ax, areaCentroids(i,1), areaCentroids(i,2), areaCentroids(i,3), ...
         ['  ' area_list{i}], 'FontSize', 8, 'Color','k', ...
         'FontWeight','bold', 'Clipping','off');
end

set(ax,'SortMethod','childorder');

%% View + export
daspect(ax,[1 1 1]);
axis(ax,'off','image');
view(ax, opts.view(1), opts.view(2));
set(fig,'renderer','opengl','Color','w');
drawnow;
exportgraphics(fig, out_filename, 'Resolution', 300, 'BackgroundColor','w');

[img,~,~] = imread(out_filename);
mask  = img(:,:,1)==255 & img(:,:,2)==255 & img(:,:,3)==255;
imwrite(img, out_filename, 'Alpha', uint8(~mask)*255);
close(fig);
end
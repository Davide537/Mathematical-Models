function [] = brain3d_conn(conn, out_filename, opts)
% conn(i,j) = weight from area j to area i, in the model's area order.
% opts (all optional): .keep_frac .log_scale .lw_range .view .roll
%                      .node_size (marker area, pt^2) .font_size .cam_push .pin_file

if nargin < 2 || isempty(out_filename)
    out_filename = sprintf('Fig/3dBrain/brain3d_conn_%s.png', datestr(now,'mmdd_HHMMSS'));
end
if nargin < 3, opts = struct(); end

def = struct('keep_frac',0.5, 'log_scale',true, 'lw_range',[0.2 1.5], ...
             'view',[-125 5], 'roll',-1.8580, 'node_size',300, 'font_size',10, ...
             'cam_push',15, 'pin_file','node_pos.mat');
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(opts, fn{k}) || isempty(opts.(fn{k})), opts.(fn{k}) = def.(fn{k}); end
end

converter = [28 25 20 90 36 74 88 71 45 21 63 50 86 72 52 75 83 51 12 16 78 84 93 69 41 27 95 89 10 77];
area_list = {'V1','V2','V4','DP','MT','8m','5','8l','2',...
             'TEO','F1','STPc','7A','46d','10','9/46v','9/46d',...
             'F5','TEpd','PBr','7m','LIP','F2','7B','ProM','STPi',...
             'F7','8b','STPr','24c'};
Nareas = numel(converter);

fig = figure('visible','off','Color','w','Position',[100 100 1000 600]);
ax = axes('Parent',fig); set(ax,'Color','w'); hold(ax,'on');

%% Node positions
if ~isempty(opts.pin_file)
    S = load(opts.pin_file);
    areaCentroids = [S.pos(:,1), -S.pos(:,2), zeros(Nareas,1)];
    opts.view = [0 90]; opts.roll = 0;
else
    load('C:\Users\david\OneDrive\Documenti\GitHub\Mathematical Models\WM Model\data\scalablebrainatlasdata.mat')
    areaCentroids = nan(Nareas,3);
    for i = 1:Nareas
        vIdx = unique(faces(find(labels==converter(i)),:));
        areaCentroids(i,:) = mean(vertices(vIdx,:),1);
    end
end

%% Edge selection (undirected: stronger direction of each pair)
C = conn; C(1:Nareas+1:end) = 0;
U = max(C, C');
[ii,jj] = find(triu(U,1) > 0);
w = U(sub2ind(size(U), ii, jj));
[w, ord] = sort(w,'ascend'); ii = ii(ord); jj = jj(ord);
nKeep = max(1, round(opts.keep_frac * numel(w)));
sel = numel(w)-nKeep+1 : numel(w);
ii = ii(sel); jj = jj(sel); w = w(sel);

if opts.log_scale, wv = log10(w); else, wv = w; end
wn = (wv - min(wv)) / max(eps, (max(wv) - min(wv)));
lw = opts.lw_range(1) + wn * diff(opts.lw_range);

cWeak   = [0.90 0.30 0.25];
cStrong = [0.82 0.10 0.08];
for e = 1:numel(w)
    P = areaCentroids([ii(e) jj(e)],:);
    col = cWeak + wn(e)*(cStrong - cWeak);
    plot3(ax, P(:,1), P(:,2), P(:,3), '-', 'LineWidth', lw(e), ...
          'Color', [col, 0.5 + 0.4*wn(e)]);
end

%% View (set before computing camera direction)
set(fig,'renderer','opengl','Color','w');
daspect(ax,[1 1 1]);
axis(ax,'off'); axis(ax,'image');
view(ax, opts.view(1), opts.view(2));
camroll(ax, opts.roll);
set(ax,'Projection','orthographic');


%% Labels: translucent white backing circle + bold text, pushed toward camera
for i = 1:Nareas
    if any(isnan(areaCentroids(i,:))), continue; end
    vOut = areaCentroids(i,:);

    text(ax, vOut(1), vOut(2), vOut(3), area_list{i}, ...
         'FontSize', opts.font_size, ...
         'Color', 'black', ...
         'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'Clipping', 'off');
end

set(ax,'SortMethod','childorder');
drawnow;

%% Export (same as brain3d)
% exportgraphics(fig, out_filename, 'Resolution', 300, 'BackgroundColor', 'w');
% [img, ~, ~] = imread(out_filename);
% mask = img(:,:,1) == 255 & img(:,:,2) == 255 & img(:,:,3) == 255;
% imwrite(img, out_filename, 'Alpha', uint8(~mask) * 255);
% close(fig);
%% Export (opaque white background)
exportgraphics(fig, out_filename, 'Resolution', 300, 'BackgroundColor', 'w');
%% Save editable .fig (same name as the PNG)
[p,n,~] = fileparts(out_filename);
set(fig, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');  % figure was created invisible; make it open visible
savefig(fig, fullfile(p, [n '.fig']));
close(fig);
end
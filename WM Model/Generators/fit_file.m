area_list = {'V1','V2','V4','DP','MT','8m','5','8l','2','TEO','F1','STPc','7A','46d','10','9/46v','9/46d', ...
             'F5','TEpd','PBr','7m','LIP','F2','7B','ProM','STPi','F7','8b','STPr','24c'};
img = imread('Generators\Screenshot 2026-11-16 at 14.06.28.png');   % use the full path if needed
figure('WindowState','maximized'); imshow(img); hold on
pos = nan(30,2);
for i = 1:30
    title(sprintf('Click the DOT of node %d/30: %s', i, area_list{i}), 'FontSize', 14);
    [x,y] = ginput(1);
    pos(i,:) = [x y];
    plot(x,y,'g+','MarkerSize',10); text(x+6,y,area_list{i},'Color','g');
end
save('node_pos.mat','pos');
disp('Saved node_pos.mat')
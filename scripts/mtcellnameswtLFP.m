% cell name list (with redundant names eliminated)
global DataPath
% run([DataPath 'SaccadeTaskList'])
cellinfo = importdata([DataPath 'cellinfo.csv']);
data=  cellinfo.data;
textdata = cellinfo.textdata;
Ncell = size(textdata,1)-1;
mtnames = cell(Ncell,1);
electrodetype = zeros(Ncell,1);
for i=1:Ncell
    mtnames{i} = textdata{i+1,1};
    electrodetype(i) = data(i,1);
end
Uprob = find(electrodetype==0);
singleelectrode = find(electrodetype==1);
UsableExp = []; UsableChan = [];
wtrepeats = find(data(:,2));
wthyperflow = find(data(:,3));
wtlaminarprobe = find(data(:,1)==0);
wttask = find(data(:,4));
hfwtlaminarprobe = intersect(wthyperflow, wtlaminarprobe);
hfwtsingleelectrode = setdiff(wthyperflow, wtlaminarprobe);
chaninfo = data(:,5:end);
for i=wthyperflow'
    goodchans = find(chaninfo(i,:)==1);
    UsableExp = [UsableExp  ones(1,length(goodchans))*i];
    UsableChan= [UsableChan goodchans];
%     fprintf(' exp name %s good chan N %d \n',mtnames{i}, length(goodchans));
end
Nunit = length(UsableExp);

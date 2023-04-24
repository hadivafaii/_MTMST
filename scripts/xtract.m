%% load designstruct files to save stim and spikes (Nardin)
clear
clc

configPath
file_names = dir(string(DataPath) + 'designstruct/');
num_files = length(file_names);

for ii=3:num_files
    name_ = file_names(ii).name;
    load_path = string(DataPath) + 'designstruct/' + name_;
    load(load_path);

    stim1 = design.stim;
    stim2 = design.stim2;
    stimR = design.stimR;
    fixlost = design.fixlost;
    fixlostR = design.fixlostR;
    badspks = design.badspikes;
    badspksR = design.badspikesR;

    expt_name = design.name;
    cellindex = design.cellindex;
    field = design.field;
    nx = design.Nx;
    ny = design.Ny;
    tres = design.tres;
    spatres = design.spatres;
    latency = design.latency;
    partition = design.partition;
    partitionR = design.partitionR;
    rf_loc = design.RFloc;

    [n_channels, n_segments] = size(design.spikes);
    assert(n_segments == length(partition) - 1)
    n_mutations = [];
    for seg=1:n_segments
        for cc=1:n_channels
            n_mutations(end+1) = length(design.spikes{cc, seg});
        end
    end
    n_mutations = max(n_mutations);
    nt = max(partition);
    spks = NaN(nt, n_channels, n_mutations);
    for seg=1:n_segments
        trange = partition(seg)+1:partition(seg + 1);
        for cc=1:n_channels
            data = design.spikes{cc, seg};
            num = size(data);
            num = num(length(num));
            for mm=1:num
                if isempty(data{1, mm})
                    continue
                end
                spks(trange, cc, mm) = data{1, mm};
            end
        end
    end

    if isempty(design.spikesR)
        spksR = design.spikesR;
    else
        ntR = length(stimR);
        spksR = zeros(ntR, num_channels);
        for cc=1:num_channels
            spksR(:, cc) = design.spikesR{cc};
        end
    end

    spkst = design.spikest;
    spkstR = design.spikestR;
    
    % HF diameter info
    diameter = [];
    for i = 1:numel(design.taskparas)
    	taskparas_contents = design.taskparas{i};
    	diameter(end+1) = taskparas_contents.maskdiameter;
    end
    assert(length(diameter) == length(partition) - 1)

    % repeat info
    diameterR = [];
    tind_start_all = [];
    fix_lost_all = [];
    psth_raw_all = [];
    repeats = design.Repeats;
    if ~isempty(repeats)
        [tind_start, spkstot, FixLostAll, psth_raw] = ExtractRepeatData(design, 1);
        nb_good_repeats = size(psth_raw, 1);
        nt_repeats = size(psth_raw, 2);

        tind_start_all = zeros(num_channels, nb_good_repeats);
        fix_lost_all = zeros(num_channels, nb_good_repeats, nt_repeats);
        psth_raw_all = zeros(num_channels, nb_good_repeats, nt_repeats);

        % get all repeat psth data
        for cc=1:num_channels
            [tind_start, spkstot, FixLost, psth_raw] = ExtractRepeatData(design, cc);
            tind_start_all(cc, :) = tind_start;
            fix_lost_all(cc, :, :) = FixLost;
            psth_raw_all(cc, :, :) = psth_raw;
        end

        % HF diameter info
        for i = 1:numel(design.taskparasR)
    	    taskparas_contents = design.taskparasR{i};
    	    diameterR(end+1) = taskparas_contents.maskdiameter;
        end
        assert(length(diameterR) == length(partitionR) - 1)
    end

    % LFP
    lfp = transpose(design.LFP);
    lfpR = transpose(design.LFPR);

    % other
    opticflows = design.opticflows;
    centerx = design.centerx;
    centery = design.centery;
    opticflowsR = design.opticflowsR;
    centerxR = design.centerxR;
    centeryR = design.centeryR;

    disp('[INFO] saving file   ')
    save_name = strcat('tres', string(tres), '_', expt_name)
    save_path = string(DataPath) + 'xtracted/' + save_name + '.mat';
    
    save(save_path,...
        'expt_name', 'cellindex', 'field', 'latency',...
        'partition', 'partitionR', 'rf_loc',...
        'stim1', 'stim2', 'stimR', 'nx', 'ny', 'tres', 'spatres',...
        'spks', 'spkst', 'spksR', 'spkstR', 'lfp', 'lfpR',...
        'fixlost', 'fixlostR', 'badspks', 'badspksR',...
        'opticflows', 'opticflowsR',...
        'centerx', 'centery',...
        'centerxR', 'centeryR',...
        'diameter', 'diameterR',...
        'repeats', 'tind_start_all', 'fix_lost_all', 'psth_raw_all')

end



%% load designstruct files to save stim and spikes (Yuwei)
clear
clc

configPath
mtcellnameswtLFP
mtcellnamewtLFP_ytu

file_names = dir(string(DataPath) + 'designstruct/');
num_files = length(file_names);

for ii=3:num_files
    name_ = file_names(ii).name;
    expt_name_ = name_(12:17);
    if contains(name_, 'tres25')
        tres_ = 'tres25';
    elseif contains(name_, 'tres5')
        tres_ = 'tres5';
    else
        tres_ = 'xxx';
    end

    load_path = string(DataPath) + 'designstruct/' + name_;
    load(load_path);

    stim1 = design.stim;
    stim2 = design.stim2;
    stimR = design.stimR;
    fixlost = design.fixlost;
    fixlostR = design.fixlostR;
    badspks = design.badspikes;
    badspksR = design.badspikesR;

    expt_name = design.name;
    cellindex = design.cellindex;
    field = design.field;
    nx = design.Nx;
    ny = design.Ny;
    spatres = design.spatres;
    latency = design.latency;
    partition = design.partition;
    partitionR = design.partitionR;
    rf_loc = design.RFloc;

    num_channels = length(design.spikes);

    nt = length(stim1);
    spks = zeros(nt, num_channels);
    for cc=1:num_channels
        spks(:, cc) = design.spikes{cc};
    end

    if isempty(design.spikesR)
        spksR = design.spikesR;
    else
        ntR = length(stimR);
        spksR = zeros(ntR, num_channels);
        for cc=1:num_channels
            spksR(:, cc) = design.spikesR{cc};
        end
    end

    spkst = design.spikest;
    spkstR = design.spikestR;

    % HF diameter info
    diameter = [];
    for i = 1:numel(design.taskparas)
    	taskparas_contents = design.taskparas{i};
    	diameter(end+1) = taskparas_contents.maskdiameter;
    end
    assert(length(diameter) == length(partition) - 1)

    % repeat info
    diameterR = [];
    tind_start_all = [];
    fix_lost_all = [];
    psth_raw_all = [];
    repeats = design.Repeats;
    if ~isempty(repeats)
        [tind_start, spkstot, FixLostAll, psth_raw] = ExtractRepeatData(design, 1);
        nb_good_repeats = size(psth_raw, 1);
        nt_repeats = size(psth_raw, 2);

        tind_start_all = zeros(num_channels, nb_good_repeats);
        fix_lost_all = zeros(num_channels, nb_good_repeats, nt_repeats);
        psth_raw_all = zeros(num_channels, nb_good_repeats, nt_repeats);

        % get all repeat psth data
        for cc=1:num_channels
            [tind_start, spkstot, FixLost, psth_raw] = ExtractRepeatData(design, cc);
            tind_start_all(cc, :) = tind_start;
            fix_lost_all(cc, :, :) = FixLost;
            psth_raw_all(cc, :, :) = psth_raw;
        end

        % HF diameter info
        for i = 1:numel(design.taskparasR)
    	    taskparas_contents = design.taskparasR{i};
    	    diameterR(end+1) = taskparas_contents.maskdiameter;
        end
        assert(length(diameterR) == length(partitionR) - 1)
    end

    % LFP
    lfp = transpose(design.LFP);
    lfpR = transpose(design.LFPR);

    % other
    opticflows = design.opticflows;
    centerx = design.centerx;
    centery = design.centery;
    opticflowsR = design.opticflowsR;
    centerxR = design.centerxR;
    centeryR = design.centeryR;

    disp('[INFO] saving file   ')
    save_name = strcat(tres_, '_', expt_name_)
    save_path = string(DataPath) + 'xtracted/' + save_name + '.mat';
    
    save(save_path,...
        'expt_name', 'cellindex', 'field', 'latency',...
        'partition', 'partitionR', 'rf_loc', 'num_channels',...
        'stim1', 'stim2', 'stimR', 'nx', 'ny', 'spatres',...
        'spks', 'spkst', 'spksR', 'spkstR', 'lfp', 'lfpR',...
        'fixlost', 'fixlostR', 'badspks', 'badspksR',...
        'opticflows', 'opticflowsR',...
        'centerx', 'centery',...
        'centerxR', 'centeryR',...
        'diameter', 'diameterR',...
        'repeats', 'tind_start_all', 'fix_lost_all', 'psth_raw_all')

end



%% Verify shared stims etc
clear
clc

configPath
mtcellnameswtLFP

file_names = dir(string(DataPath) + 'hadi_processed_final/');
num_files = length(file_names);

nc_t = 0;

for ii=3:num_files
    name_ = file_names(ii).name;

    load_path = string(DataPath) + 'hadi_processed_final/' + name_;
    load(load_path);

    fprintf('%d ... [INFO] file  %s ', ii, name_)
    size(spks)
    size(spksR)

    nc_t = nc_t + size(spks, 2);
end




%% extract useful info

clear
clc

configPath
mtcellnameswtLFP

file_names = dir(string(RootPath) + 'MTproject_result/NLmod');
num_files = length(file_names);

expt_channel_names = [];
channels = [];
xvs = [];
thetas = [];

ws = [];
tk = [];

for ii=1:num_files
    name_ = file_names(ii).name;

    if contains(name_, 'NLfitEonlymod.mat')
        load_path = string(RootPath) + 'MTproject_result/NLmod/' + name_;
        load(load_path)

        clean_name = erase(name_(6:end), "unit1NLfitEonlymod.mat");
        expt_name = clean_name(1:6);
        channel = str2double(erase(erase(clean_name, expt_name), "chan"));

        xv = NLfitexc.xLLdiff;
        theta = mean(NLfitexc.mods.angles);

        fprintf('[INFO]   %s + %d   has XV:  %d \n', expt_name, channel, xv)

        expt_channel_names = [expt_channel_names; expt_name];
        channels = [channels; channel];
        xvs = [xvs; xv];
        thetas = [thetas; theta];

        ws = [ws; reshape(NLfitexc.mods(1, 1).weight, 1, 225)];
        tk = [tk; reshape(NLfitexc.mods(1, 1).temporalk, 1, 40)];
    end
end

save_path = string(RootPath) + 'USEFUL_INFO.mat';
save(save_path, 'expt_channel_names', 'channels', 'xvs', 'thetas', 'ws', 'tk')




%% Plot RFs

clear
clc

configPath
mtcellnameswtLFP

file_names = dir(string(RootPath) + 'MTproject_result/NLmod');
num_files = length(file_names);

for ii=1:num_files
    name_ = file_names(ii).name;

    if contains(name_, 'NLfitEonlymod.mat')
        load_path = string(RootPath) + 'MTproject_result/NLmod/' + name_;
        load(load_path)

        clean_name = erase(name_(6:end), "unit1NLfitEonlymod.mat");
        expt_name = clean_name(1:6);
        channel = str2double(erase(erase(clean_name, expt_name), "chan"));

        xv = NLfitexc.xLLdiff;

        fprintf('[INFO]   %s + %d   has XV:  %d \n', expt_name, channel, xv)

        % plot RF
        figure; set(gcf,'Visible', 'off');
        set(gcf, 'Position',  [50, 50, 650, 200]);

        ws = reshape(NLfitexc.mods(1, 1).weight, 15, 15); subplot(1,2,1); imagesc(ws); colorbar;
        tk = NLfitexc.mods(1, 1).temporalk; subplot(122); plot(tk)

        title(xv)

        saveas(gcf, strcat('/home/hadivafa/Dropbox/slide_material/MATLAB_RFs/', clean_name, '.pdf'));
        close;
    end
end




%%

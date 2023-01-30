function [tind_start, spkstot, FixLostAll, psth_raw] = ...
        ExtractRepeatData(design, targetchan, MaxLostFrac)
% process repeat data
% tind_start: start bin of each trial at time resolution of design.tres
% spkstot: a collection of spike times, separated by -1's
% psth_raw: Trials x Times binned spikes at design.tres resolution 
% FixLostAll: binary matrix indicating fixation lost times

% exclude trials with too much fixation lost
if nargin<3
    MaxLostFrac = 0.4; % percent of allowed fixation lost
end

FixLostAll = [];
spkstot = [];
RateAvgALL = [];

psth_raw = [];
tind_start = [];
GoodRepeatTot = 0;
randomSeed = [];
Nblock = length(design.taskparasR);
stimR = design.stimR;
% markers for each block for stimulus and spikes
partitionstim = design.partitionR;
partitionspk = find(design.spikestR{targetchan}==-1);
partitionspk = [1; partitionspk];
RepeatTot = 0;


for i=1:Nblock
    % spikes for the current block
    spkst = design.spikestR{targetchan}(partitionspk(i):partitionspk(i+1)-1);
    
    taskpara = design.taskparasR{i};
    frameT = 1/taskpara.framerate;
    
    % start time of the first repeat trial
    shiftT = taskpara.preInterval + taskpara.interInterval;
    % start-end time for each repeat
    RepeatT = design.Repeats*frameT + shiftT/1000;

    randomSeed(i)=taskpara.randomSeed;
    %         keyboard
    RepeatN = taskpara.blocks;
    fprintf(' Block %d Repeat N %d Seed %d Aperture %d \n',...
        i,RepeatN,taskpara.randomSeed,taskpara.patchRadiusDeg)
    

    assert(taskpara.blocks==length(design.Repeats)-1)
    
    % number of frames for each repeat segment at monitor resolution
    frameN = unique( design.Repeats(2:end)-design.Repeats(1:end-1) );
    assert(length(frameN)==1);    
    
    % number of frames at the current temporal resolution (design.tres)    
    frameTdown = design.tres/1000;
    frameNdown = round(frameN*frameT/frameTdown);
    stimRepeat = stimR(1:frameNdown,:);
    
    
    % figure,
    % opticflows = design.opticflowsR(1:frameN,1:6) ./ (ones(frameN,1)*std(design.opticflowsR));
    % subplot(6,1,1);plot((0:frameN-1)*frameT,opticflows(:,1));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);set(gca,'Xtick',[]);
    % subplot(6,1,2);plot((0:frameN-1)*frameT,opticflows(:,2));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);set(gca,'Xtick',[]);
    % subplot(6,1,3);plot((0:frameN-1)*frameT,opticflows(:,3));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);set(gca,'Xtick',[]);
    % subplot(6,1,4);plot((0:frameN-1)*frameT,opticflows(:,4));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);set(gca,'Xtick',[]);
    % subplot(6,1,5);plot((0:frameN-1)*frameT,opticflows(:,5));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);set(gca,'Xtick',[]);
    % subplot(6,1,6);plot((0:frameN-1)*frameT,opticflows(:,6));line([0 (frameN-1)*frameT],[0 0],'LineStyle','--');axis([0 (frameN-1)*frameT -4 4]);
    
    
    % partition spike train into trials
    GoodRptTrials = 0; % number of good trials
    spks = []; % cat all spikes here
    spknum = []; % number of spikes per trial
    for r=1:RepeatN-1
        % start/end time for the current trial
        tstart = (RepeatT(r));
        tend =  (RepeatT(r+1));
        
        % indices for this trial at frame rate
        trange = ((r-1)*frameN-1+(1:frameN))*frameT;
        
        % indices for this trial at resolution "tres"
        tinds2 = round(tstart*1000/design.tres)+(1:frameNdown);
%         tstart = tinds2(1)*design.tres/1000-frameTdown;
        tinds2 = tinds2+partitionstim(i);
        tinds2 = tinds2(tinds2<length(design.fixlostR));        
        
        % spikes in the current trial
        spksR = spkst(spkst>tstart & spkst<tend);
        spksR = spksR - tstart; % ALIGN SPIKES TO THE START OF THE TRIAL
%         fprintf('Repeat %d Spike # %d \n',r,length(spksR));
        FixLost = design.fixlostR(tinds2);
        
        if sum(design.fixlostR(tinds2))<=MaxLostFrac*length(tinds2) % && ~isempty(spksR) 
            FixLostAll = [FixLostAll; FixLost(:)'];
            Ratei = spks2Rate(spksR,design.tres/1000,frameNdown)';
            psth_raw = [psth_raw; reshape(Ratei,1,[])];
            spknum(end+1) = length(spksR);
            spks = [spks; [spksR; -1]];
            tind_start(end+1) = tinds2(1);
            GoodRptTrials = GoodRptTrials+1;            
        end
    end
    
%     if isempty(spks) % no trials available for current block
%         continue;
%     end
    spkstot = [spkstot; spks];
    Rate = mean(psth_raw,1);
    GoodRepeatTot = GoodRepeatTot + GoodRptTrials;
    RepeatTot = RepeatTot+RepeatN;
    if ~isempty(RateAvgALL)
        RateAvgALL = RateAvgALL+Rate;
    else
        RateAvgALL = Rate;
    end
    %         figure(12),
    %         stimexample = design.stimR(1:frameN, :); %stimexample(:,end/2+1:end) = -stimexample(:,end/2+1:end);
    %         subplot(4,4,1);PlotSpatialk(stimexample(75,:), sqrt(NS/2), 1./design.spatres);
    %         subplot(4,4,2);PlotSpatialk(stimexample(150,:), sqrt(NS/2), 1./design.spatres);
    %         subplot(4,4,3);PlotSpatialk(stimexample(225,:), sqrt(NS/2), 1./design.spatres);
    %         subplot(4,4,4);PlotSpatialk(stimexample(300,:), sqrt(NS/2), 1./design.spatres);
    
    %         frac = 5;
    %         RateAvg = DownSampling(Rate,frac);
    %         RateAvg = RateAvg/(frameT);
end
RptTrialT = frameN*frameT;
fprintf(' Repeat Trials Length %2.3f sec \n',RptTrialT)
fprintf(' N Rpt Trial %d N Good Rpt Trials %d \n',RepeatTot,GoodRepeatTot)

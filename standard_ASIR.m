function [E_target_state_MC]=standard_ASIR(Np,initx,Re_x,Re_y,numX,numY,Total_time,xy_data,Sigma_noise,A,F,Q)

format long;

%% ======The initialization of the basic parameters=====

T_step=1;          % The size of the time cell:Time_step
q1=0.002;          % q1,q2 the level of process noise in target motion

%% ---------- initial distribution of target state
delta_p=0.1;  % the stardard deviation of the inital distribution of position
new_velocity_Variance=0.01;             % standard deviation;
% q2=2;                  % q2 the measurement noise

% Target_number = 1;
E_target_state_MC=zeros(7,Total_time);
% single_run_time=zeros(Total_time);

%% ===================== Particle filtering =========================

%% ============== PF implementation ================
Pre_T_particle=zeros(7,Total_time,Np);            % Pre_track particles
Pre_track_Z=zeros(1,Np);
w=zeros(1,Np);

% Pre_weight0=zeros(Np,Total_time);               % Particles weights of Pre-track PF
% Pre_weight=zeros(Np,Total_time);                % Normalized Particles weights
% Pre_w_likehood_all=zeros(Np,Total_time);        % likehood part of the pre-PF weights
% Pre_track_bias=zeros(Np,Total_time);            % weight bias part of the pre-PF weights


for t = 1:Total_time
    %     display(['Np=',num2str(Np),'; t=',num2str(t)]);
    %singerun_start=tic;
    %% --------------- detection procedure ----------------
    Detection_frame=xy_data(:,:,t);
    %     clean_Frame = zeros(size(Detection_frame));
    if t==1
        index_x=initx(1)/Re_x;
        index_y=initx(3)/Re_y;
        index_vx=initx(2)/Re_x;
        index_vy=initx(4)/Re_y;
        % -------generate the new partitions of particles
        %--------generate position based the detection measurements
        position_x_p=repmat(index_x,Np,1)+delta_p*randn(Np,1);
        position_y_p=repmat(index_y,Np,1)+delta_p*randn(Np,1);
        %% --------初始粒子均匀分布
        %         position_x_p = random('unif',1,10,Np,1);
        %         position_y_p = random('unif',1,10,Np,1);
        %% --------generate velocity based on the detections
        velocity_x_p=repmat(index_vx,Np,1);
        velocity_y_p=repmat(index_vy,Np,1);
        %        velocity_x_p=random('unif',0.5,1.5,Np,1);;
        %        velocity_y_p=random('unif',0.5,1.5,Np,1);;
        %--------generate velocity variance
        velocity_p_kk1=new_velocity_Variance.*ones(Np,1);
        
        %--------new_pretrack=zeros(4,Num_n_target,Np);
        
        %         Pre_T_life_quality=ones(1,Np);
        %% initialization(if t=1)
        for i=1:Np
            Pre_T_particle(1:6,t,i)=[position_x_p(i);velocity_x_p(i);velocity_p_kk1(i);position_y_p(i);velocity_y_p(i);velocity_p_kk1(i)];
            
            %initial weights
            Z_x_index=ceil(Pre_T_particle(1,t,i));
            Z_y_index=ceil(Pre_T_particle(4,t,i));
            if Z_x_index<=numX && Z_x_index>0 && Z_y_index<=numY && Z_y_index>0
                Pre_track_Z(i)=Detection_frame(Z_y_index,Z_x_index);
                Pre_T_particle(7,t,i)=Detection_frame(Z_y_index,Z_x_index); %该粒子（样本）处的观测值
                %% Gaussian likelihood ratio
                w(i)=exp(0.5*(2*Detection_frame(Z_y_index,Z_x_index)*A-A^2));
            else
                w(i)=0;
            end
            w=w./sum(w);
        end
        Pre_T_particle(7,t,:)=1;
        
        
    else %when t>1
        %% PF iteration
        %% === sample index funciton: Resampling
        
        [index_sample]=Sample_index(w); %只要i_j
        
        for j=1:Np % 6%
            
            Pre_T_particle(1:6,t,j)= sample_KP(Pre_T_particle(1:6,t-1,index_sample(j)),F,Q); %propagate using KP as in Standard SIR
            %assign weights
            Z_x_index=ceil(Pre_T_particle(1,t,j));
            Z_y_index=ceil(Pre_T_particle(4,t,j));
            if Z_x_index<=numX && Z_x_index>0 && Z_y_index<=numY && Z_y_index>0
                Pre_track_Z(j)=Detection_frame(Z_y_index,Z_x_index);
                %                 Pre_T_life_quality(j)=Pre_T_life_quality(j)+Detection_frame(Z_y_index,Z_x_index);
                Pre_T_particle(7,t,j)=Detection_frame(Z_y_index,Z_x_index); %该粒子（样本）处的观测值
                % Gaussian likelihood ratio
                w(j)=w(j)*(exp(0.5*(2*Detection_frame(Z_y_index,Z_x_index)*A-A^2)))/w(index_sample(j)); % 由于不是SIR每一步重采样的1/N权重,故上一步权重不可省略。
            else
                w(j)=0;
            end
        end
        
        w=w./sum(w);
        
        %% === retain the bias of sample: the likehood
        %particle_likehood_after_bias=w(index_sample);
        
    end
    
end
%% record the estimates
E_target_state=mean(Pre_T_particle(:,:,:),3); %对A的第dim维度求均值
E_target_state_MC([1,2],:)=E_target_state([1,2],:)*Re_x;
E_target_state_MC([4,5],:)=E_target_state([4,5],:)*Re_y;
E_target_state_MC([3,6,7],:)=E_target_state([3,6,7],:);







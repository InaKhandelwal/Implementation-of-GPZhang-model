tic;
clc;
clear all;
close all;
filename='sunspot.xlsx';

data=xlsread(filename);
%plot(data);
data_size=size(data,1);
test_size=67;
train_size=data_size-test_size;
train_d=data(1:train_size);
test_d=data(train_size+1:data_size);

MinMSE=inf;

%[norm_data,PS] = mapminmax(data',0,1);
%%------------- for training --------------------------------------
model=arima(9,0,0);
fit = estimate(model,train_d);

ar_coeffs=(fit.AR)';
fi=cell2mat(ar_coeffs);

ma_coeffs=(fit.MA)';
theta=cell2mat(ma_coeffs);

dif=fit.D;

no_parameters=size(fi)+size(theta)+dif;


no_itr=1;

%%------------- for training------------------------------
for p=1:no_itr
    sigma=sqrt(fit.Variance);
    
    wn=sigma*randn(data_size,1);
    
    Y_train((1:(train_size-no_parameters)),1)=train_d((1+no_parameters:train_size),1);
    % for l=(1+size(fi)+size(theta)):(train_size-size(fi)-size(theta))
    for l=(1+no_parameters):(data_size)
        t1=fit.Constant;
        ar_sum=0;
        for i=1:size(fi)
            ar_term(i)=(fi(i)*data(l-i));
            
            ar_sum=ar_sum + ar_term(i);
            
        end
        
        ma_sum=0;
        for j=1:size(theta)
            ma_term(j)=theta(j)*wn(l-j);
            ma_sum=ma_sum + ma_term(j);
        end
        if l<=train_size
            L_train((l-no_parameters):(train_size-(no_parameters)))=t1+ar_sum + ma_sum+wn(l);
        else
            L_test(l-train_size:test_size)=t1+ar_sum + ma_sum+wn(l);
        end
        % t_for(l-size(fi)-size(theta))=t_sum(l);
    end
    Res=Y_train-L_train';
    %%---------------  for testing --------------------------------------
    %for r=1:(size(fi)+size(theta)+size(dif))
    T_Residual=zeros(no_parameters,1);
    Residual=[T_Residual;Res];
    
    Res_data=[Residual;test_d];
    
    %%-----------------for forecasting the residuals with ANN------------------
    
    %inpt_nodes=4;hdn_nodes=4;
    diff_td=diff(Res_data);
    sigma=std(diff_td);
    wn=sigma*randn(data_size,1);
    s_res=size(Residual,1);
    train_res=Residual(1:s_res);
    max_hidden=12;
    max_input=12;
    for a=3:max_hidden
        for b=3:max_input
            inpt_nodes=b;
            hdn_nodes=a;
            
            train_lim=s_res-inpt_nodes;
            test_lim=test_size;
            p_train=[];t_train=[];
            p_test=[];t_test=[];
            
            
            
            for i=1:train_lim
                p_train=[p_train Residual(i:i+(inpt_nodes-1))];
                t_train=[t_train Residual(i+inpt_nodes)];
            end
            for i=1:test_lim
                if i<=inpt_nodes
                    p_test=[p_test Res_data((s_res-inpt_nodes+i):((s_res-inpt_nodes+i)+inpt_nodes-1))];
                    t_test=[t_test Res_data((s_res-inpt_nodes+i)+inpt_nodes)];
                else
                    i=i-inpt_nodes;
                    p_test=[p_test test_d(i:i+(inpt_nodes-1))];
                    t_test=[t_test test_d(i+inpt_nodes)];
                end
            end
            %t_test=D_aorg(s_train+1:s_org);
            
            
            net=newff(p_train, t_train,hdn_nodes,{'logsig','purelin'},'trainlm','learngdm','mse');
            %net.trainParam.epochs = 2000;
            net.trainParam.show = 1000;
            net.trainParam.showCommandLine=0;
            net.trainParam.showWindow=0;
            net.trainParam.lr = 0.05;
            % net.trainParam.mc = 0.9;
            % net.trainParam.goal = 1e-5;
            
            [net,tr]=train(net,p_train,t_train);
            
            T=p_train(:,train_lim);
            T(1:end-1)=T(2:end);
            T(end)=t_train(end);
            t_for=sim(net,p_test);
            t_for=zeros(test_lim,1);
            for i=1:test_lim
                t_for(i)=sim(net,T)+wn(size(Res_data,1)-test_lim+1);
                T(1:end-1)=T(2:end);
                T(end)=t_for(i);
                lt_for(i)=t_for(i);
            end
            
            final_forecast=t_for+L_test';
            
            %calculating error measures
            [ MSE,MAE,MAPE,RMSE,SSE,MPE]=errorfunction(test_d,final_forecast,test_size);
            
            if MSE < MinMSE
                MinMSE = MSE;
                MinMAE = MAE;
                MinMAPE = MAPE;
                MinRMSE=RMSE;
                MinSSE = SSE;
                MinMPE=MPE;
                InputNode = inpt_nodes;
                HiddenNode = hdn_nodes;
                predicted=final_forecast;
                MinIttNo =p;
            end
           
        end   %for input nodes
    end      %for hidden nodes
end      %for iterations

fprintf('Minimum MSE=%f\nOccured at iteration=%d\n',MinMSE,MinIttNo);
fprintf('Minimum MAE=%f\n',MinMAE);
fprintf('Minimum MAPE=%f\n',MinMAPE);
fprintf('Minimum RMSE=%f\n',MinRMSE);
fprintf('Minimum SSE=%f\n',MinSSE);
fprintf('Minimum MPE=%f\n',MinMPE);


x=1:size(test_d);
plot(x,test_d,'k-',x,predicted,'r:')

toc;

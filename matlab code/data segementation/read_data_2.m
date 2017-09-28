function [csi,fname]=read_data_2(url)
%csi为去噪、pca、dwt之后的前4列主元素，fname是标签序号。
cd (url);
file=struct2cell(dir);
name={file{1,3:end}};
csi={};
fname={};
t=1;
a_num=0;
[a_1,b_1]=butter(2,0.06,'low');
for i=1:size(name,2)
    filename=name{i};
    data=get_csi_data(filename);
    a=size(data,1);   
    if(isempty(data)==0 && a>2000) 
        a_num=[a_num,a];  
        c_data=filter(a_1,b_1,data);
        csi{t}=c_data;        
        fname{t}={filename};
        t=t+1;
    end
end
% a_num=sort(a_num);
% choose_num=a_num(2);

for ii=1:size(csi,2)
%     csi{ii}=csi{ii}(1:choose_num,:,:);
    data_1=pca(squeeze(csi{ii}(:,1,:))');
    data_2=pca(squeeze(csi{ii}(:,2,:))');
    data_3=pca(squeeze(csi{ii}(:,3,:))');
    data_4=[data_1(:,1:4),data_2(:,1:4),data_3(:,1:4)];
    dat2=[];
    for tt=1:size(data_4,2)
        dat1=dwt(data_4(:,tt),'db4');
        dat2=[dat2,dat1];
    end
    csi{ii}=dat2(:,2:end);
end
  







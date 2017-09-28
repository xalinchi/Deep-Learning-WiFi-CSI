function distance= dis(data1,data2)
%º∆À„dtwæ‡¿Î
% distance=[];
for ii=1:size(data1,2)
    for jj=1:size(data1{1,ii},2)
        num1=data1{1,ii};
        num2=data2{1,1};
        di(jj)=dtw(num1(:,jj)',num2(:,jj)');
    end
    distance(ii)=sum(di);
end



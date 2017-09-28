function csi_array=get_csi_data(url)
csi_trace = read_bf_file(url);
[csi_size,tmp]=size(csi_trace);

samples=zeros(csi_size,3,30);
for ii=1:csi_size
    csi_entry = csi_trace{ii};
    csi_s=size(csi_entry);
    if(csi_s(1)==0)
        csi_size=ii;
        break;
    end
    csi = get_scaled_csi(csi_entry);
    %csi=filter(myfile,csi);
    csi=db(abs(squeeze(csi(1,:,:)).'));
    csi=csi';
    samples(ii,1,:)=csi(1,:);
    samples(ii,2,:)=csi(2,:);
    samples(ii,3,:)=csi(3,:);
end

for ii=1:csi_size
    for jj=1:30
        for kk=1:3
            if samples(ii, kk, jj) < -20
                samples(ii, kk, jj) = 0;
            else if samples(ii, kk, jj) > 50
                    samples(ii, kk, jj) = 25;
                end
            end
        end
    end
end

csi_array=samples(1:csi_size,:,:);
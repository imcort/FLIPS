if exist("visadevObj")

else
    visadevObj = visadev("TCPIP0::192.168.5.102::inst0::INSTR");
end

freq_str = query(visadevObj, 'CALC1:DATA:STIM?');
frequencies = str2num(freq_str);

t = 600;

rec = zeros(t,length(frequencies));

for i=1:t
    visadev_data = writeread(visadevObj,"CALC1:DATA? FDATA");
    data = str2num(visadev_data);
    rec(i,:) = data;
    plot(frequencies,data)
    drawnow
end

filename = datestr(datetime('now'),'yyyy-mm-dd_HH-MM-SS');
csvwrite([filename '.csv'], [frequencies' rec']);

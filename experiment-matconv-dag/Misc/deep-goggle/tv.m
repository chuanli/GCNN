function [e, dx] = tv(x,beta)
% --------------------------------------------------------------------
if(~exist('beta', 'var'))
  beta = 1; % the power to which the TV norm is raized
end
d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
v = sqrt(d1.*d1 + d2.*d2).^beta ;
e = sum(sum(sum(sum(v)))) ;
if nargout > 1
  d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
  d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
  d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
  d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
  d11(:,1,:,:) = - d1_(:,1,:,:) ;
  d22(1,:,:,:) = - d2_(1,:,:,:) ;
  dx = beta*(d11 + d22);
  if(any(isnan(dx)))
  end
end
#ifndef GUARD_bd_h
#define GUARD_bd_h

#include "rn.h"
#include "info.h"
#include "tree.h"

bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, rn& gen);
bool bdc(tree& x, xinfo& xi, dinfo& di, pinfo& pi, rn& gen, const std::vector<double>& latent_nu, std::vector<bool>& monotone_flags, bool data_aug);


#endif

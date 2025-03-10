#pragma once
#include "lut.h"

namespace gt {

struct InternalPower {

  std::string related_pin;

  std::optional<Lut> rise_power;
  std::optional<Lut> fall_power;

  void scale_time(float s);
  void scale_capacitance(float s);

  std::optional<float> power(Tran, Tran, float, float) const;
};

std::ostream& operator << (std::ostream&, const InternalPower&);

};  // namespace gt

import { Thermometer, Droplets, CloudRain } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface EnvironmentalData {
  temperature: string;
  humidity: string;
  rainfall: string;
}

interface EnvironmentalInputsProps {
  data: EnvironmentalData;
  onChange: (data: EnvironmentalData) => void;
}

const EnvironmentalInputs = ({ data, onChange }: EnvironmentalInputsProps) => {
  const handleChange = (field: keyof EnvironmentalData, value: string) => {
    onChange({ ...data, [field]: value });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <CloudRain className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-medium text-foreground">
          Environmental Data
          <span className="text-muted-foreground font-normal ml-2">(Optional)</span>
        </h3>
      </div>

      <div className="grid sm:grid-cols-3 gap-4">
        {/* Temperature */}
        <div className="space-y-2">
          <Label htmlFor="temperature" className="flex items-center gap-2 text-sm">
            <Thermometer className="h-4 w-4 text-tomato" />
            Temperature
          </Label>
          <div className="relative">
            <Input
              id="temperature"
              type="number"
              placeholder="25"
              value={data.temperature}
              onChange={(e) => handleChange("temperature", e.target.value)}
              className="pr-10"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
              °C
            </span>
          </div>
        </div>

        {/* Humidity */}
        <div className="space-y-2">
          <Label htmlFor="humidity" className="flex items-center gap-2 text-sm">
            <Droplets className="h-4 w-4 text-accent" />
            Humidity
          </Label>
          <div className="relative">
            <Input
              id="humidity"
              type="number"
              placeholder="60"
              value={data.humidity}
              onChange={(e) => handleChange("humidity", e.target.value)}
              className="pr-10"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
              %
            </span>
          </div>
        </div>

        {/* Rainfall */}
        <div className="space-y-2">
          <Label htmlFor="rainfall" className="flex items-center gap-2 text-sm">
            <CloudRain className="h-4 w-4 text-primary" />
            Rainfall
          </Label>
          <div className="relative">
            <Input
              id="rainfall"
              type="number"
              placeholder="10"
              value={data.rainfall}
              onChange={(e) => handleChange("rainfall", e.target.value)}
              className="pr-10"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
              mm
            </span>
          </div>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        Adding environmental conditions can improve prediction accuracy by considering factors that influence disease development.
      </p>
    </div>
  );
};

export default EnvironmentalInputs;

import React, { useId, useState } from "react";
import { Eye, EyeOff } from "lucide-react";

type Props = Omit<React.InputHTMLAttributes<HTMLInputElement>, "type" | "id"> & { label: string };

export function PasswordField({ label, ...props }: Props) {
  const id = useId();
  const [visible, setVisible] = useState(false);
  const action = visible ? "Masquer le mot de passe" : "Afficher le mot de passe";
  return <div className="password-field">
    <label htmlFor={id}>{label}</label>
    <div className="password-input-wrap">
      <input {...props} id={id} type={visible ? "text" : "password"} />
      <button type="button" className="password-toggle" aria-label={`${action} (${label})`}
        title={action} aria-controls={id} aria-pressed={visible} disabled={props.disabled}
        onClick={() => setVisible(current => !current)}>
        {visible ? <EyeOff size={20} /> : <Eye size={20} />}
      </button>
    </div>
  </div>;
}

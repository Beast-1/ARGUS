import { apiFileUrl } from "../api/client";
import "./AssetThumbnail.css";

interface AssetThumbnailProps {
  previewUrl: string | null;
  alt: string;
  liveLabel?: string;
}

export function AssetThumbnail({ previewUrl, alt, liveLabel }: AssetThumbnailProps) {
  return (
    <div className="asset-thumb">
      {liveLabel && (
        <div className="asset-thumb-live">
          <i />
          {liveLabel}
        </div>
      )}
      {previewUrl ? (
        <img className="asset-thumb-img" src={apiFileUrl(previewUrl)} alt={alt} />
      ) : (
        <div className="asset-thumb-empty">no preview</div>
      )}
    </div>
  );
}

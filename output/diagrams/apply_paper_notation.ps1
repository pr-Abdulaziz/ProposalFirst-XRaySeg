param(
    [string]$Source = (Join-Path $PSScriptRoot 'proposal-first-pidray-occ-hybrid.drawio'),
    [string]$Destination = (Join-Path $PSScriptRoot 'proposal-first-pidray-occ-hybrid.drawio')
)

$ErrorActionPreference = 'Stop'
[xml]$doc = Get-Content -LiteralPath $Source -Raw -Encoding UTF8
$root = $doc.mxfile.diagram.mxGraphModel.root

function Get-Cell([string]$id) {
    $cell = $root.SelectSingleNode("mxCell[@id='$id']")
    if (-not $cell) { throw "Missing mxCell: $id" }
    return $cell
}

function Set-Value([string]$id, [string]$value) {
    (Get-Cell $id).SetAttribute('value', $value)
}

# PIDray paper aggregation and released-code input are shown together.
$r0Tokens = Get-Cell 'r0-tokens'
$r0Tokens.SetAttribute(
    'value',
    '<b>Fᵢ⁰</b><br>paper: Agg(Fᵢ¹…Fᵢ⁵)<br>code: Fᵢ⁵ · +PE'
)
$r0Tokens.SetAttribute(
    'style',
    ([string]$r0Tokens.style).Replace('fontSize=12;', 'fontSize=9;')
)

# Main bilayer lane: use the Occ paper symbols.
Set-Value 'base-tokens' 't<sub>mask</sub> · t<sub>IoU</sub>'
Set-Value 'overlap' 'm̂ᵢⱼᵉ<br>overlap mask<br>1×128×128<br><b>auxiliary only</b>'
Set-Value 'ir' 'fᵢⁱᵐᵍ (+d∅)<br>256×32×32'

$trCell = Get-Cell 'tr'
$trCell.SetAttribute(
    'value',
    'T<sub>ij</sub><sup>r</sup>=Cat(t<sub>mask</sub>,t<sub>IoU</sub>,<br>fᵢⱼˢᵖᵃʳˢᵉ)<br>1×10×256'
)
$trCell.SetAttribute(
    'style',
    ([string]$trCell.style).Replace('fontSize=11;', 'fontSize=9;')
)

Set-Value 'returns' 'f<sub>r,ij</sub><sup>img</sup><br>256×32×32'
Set-Value 'token-returns' 'f<sub>r,ij</sub><sup>sparse</sup><br>1×10×256'
Set-Value 'ie' 'f<sub>e,ij</sub><sup>img</sup> = fᵢⁱᵐᵍ + f<sub>r,ij</sub><sup>img</sup><br>256×32×32'
Set-Value 'te' 'T<sub>ij</sub><sup>e</sup>=Cat(t<sub>mask</sub>,t<sub>IoU</sub>,<br>f<sub>e,ij</sub><sup>sparse</sup>)<br>1×10×256'

# Detail panel: paper equation first, exact local token packing second.
Set-Value 'd-ir' '<b>Dense input</b><br>fᵢⁱᵐᵍ (+d∅)<br>256×32×32'

$detailTokens = Get-Cell 'd-tr'
$detailTokens.SetAttribute(
    'value',
    '<b>T<sub>ij</sub><sup>r</sup></b>=Cat(t<sub>mask</sub>,t<sub>IoU</sub>,fᵢⱼˢᵖᵃʳˢᵉ)<br>code packs [1 IoU, 4 mask, 5 prompt]<br>1×10×256'
)
$detailTokens.SetAttribute(
    'style',
    ([string]$detailTokens.style).Replace('fontSize=11;', 'fontSize=9;')
)

Set-Value 'd-ie' 'f<sub>e,ij</sub><sup>img</sup>=fᵢⁱᵐᵍ+f<sub>r,ij</sub><sup>img</sup>'
Set-Value 'd-te' 'T<sub>ij</sub><sup>e</sup>=Cat(t<sub>mask</sub>,t<sub>IoU</sub>,f<sub>e,ij</sub><sup>sparse</sup>)'
Set-Value 'e-phir-imgplus2' 'fᵣⁱᵐᵍ'
Set-Value 'e-phir-tokplus2' 'fᵣˢᵖᵃʳˢᵉ'

Set-Value 'source-note' 'Blue/orange: PIDray · purple/green: Occ bilayer SAM · proposal gate and detector-to-prompt splice: proposed · main symbols follow the papers; code-only packing is labeled explicitly · one retained RoI j shown.'

$doc.mxfile.SetAttribute('modified', '2026-08-06T00:00:00.000Z')
[System.IO.File]::WriteAllText(
    $Destination,
    $doc.OuterXml,
    [System.Text.UTF8Encoding]::new($false)
)

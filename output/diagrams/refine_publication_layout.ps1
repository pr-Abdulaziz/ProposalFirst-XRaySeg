param(
    [string]$Source = (Join-Path $PSScriptRoot 'proposal-first-pidray-occ-hybrid-publication.drawio'),
    [string]$Destination = (Join-Path $PSScriptRoot 'proposal-first-pidray-occ-hybrid-clean.drawio')
)

$ErrorActionPreference = 'Stop'
[xml]$doc = Get-Content -LiteralPath $Source -Raw -Encoding UTF8
$root = $doc.mxfile.diagram.mxGraphModel.root

function Get-Cell([string]$id) {
    $cell = $root.SelectSingleNode("mxCell[@id='$id']")
    if (-not $cell) { throw "Missing mxCell: $id" }
    return $cell
}

function Set-Geometry([string]$id, [double]$x, [double]$y, [double]$width, [double]$height) {
    $cell = Get-Cell $id
    $geometry = $cell.SelectSingleNode('mxGeometry')
    $geometry.SetAttribute('x', [string]$x)
    $geometry.SetAttribute('y', [string]$y)
    $geometry.SetAttribute('width', [string]$width)
    $geometry.SetAttribute('height', [string]$height)
}

function Set-OrthogonalEdge([string]$id, [object[]]$points = @()) {
    $cell = Get-Cell $id
    $style = [string]$cell.style
    $style = $style.Replace('edgeStyle=none;curved=1;', 'edgeStyle=orthogonalEdgeStyle;rounded=1;curved=0;')
    if ($style -notmatch 'jumpStyle=') { $style += 'jumpStyle=arc;jumpSize=8;' }
    $cell.SetAttribute('style', $style)

    $geometry = $cell.SelectSingleNode('mxGeometry')
    $old = $geometry.SelectSingleNode("Array[@as='points']")
    if ($old) { [void]$geometry.RemoveChild($old) }

    if ($points.Count -gt 0) {
        $array = $doc.CreateElement('Array')
        $array.SetAttribute('as', 'points')
        foreach ($point in $points) {
            $mxPoint = $doc.CreateElement('mxPoint')
            $mxPoint.SetAttribute('x', [string]$point[0])
            $mxPoint.SetAttribute('y', [string]$point[1])
            [void]$array.AppendChild($mxPoint)
        }
        [void]$geometry.AppendChild($array)
    }
}

# Draw.io treats the unescaped semicolon as a style separator. Encoding it makes
# the embedded PNG render while keeping the diagram fully portable.
$imageCell = Get-Cell 'input'
$imageCell.SetAttribute('style', ([string]$imageCell.style).Replace('data:image/png;base64,', 'data:image/png%3Bbase64,'))

# Detection outputs sit immediately above and below their producer. This removes
# the two long blue arcs that crossed the gate and prompt bridge.
Set-Geometry 'out-class' 2425 292 175 48
Set-Geometry 'out-box'   2620 292 175 48
Set-OrthogonalEdge 'e-class'
Set-OrthogonalEdge 'e-box'

# Fixed routing corridors for the main architecture. Most connectors are now
# orthogonal and rounded; only the short R0 control arc remains curved.
Set-OrthogonalEdge 'e-input-sam' @(@(150, 650), @(230, 650), @(230, 851))
Set-OrthogonalEdge 'e-fpn-rpn'   @(@(1090, 223), @(1090, 442))
Set-OrthogonalEdge 'e-fpn-bus'   @(@(1100, 279), @(1100, 563), @(1735, 563))
Set-OrthogonalEdge 'e-phi-gate' @(@(2180, 245), @(2880, 245))
Set-OrthogonalEdge 'e-sparse-cat' @(@(3738, 665), @(1395, 665), @(1395, 1105), @(906, 1105))

Set-Geometry 'phir' 1440 790 245 240
Set-OrthogonalEdge 'e-ir-phir'
Set-OrthogonalEdge 'e-tr-phir'
Set-OrthogonalEdge 'e-pe-phir'
Set-OrthogonalEdge 'e-ir-plus-img' @(@(1050, 765), @(2079, 765))
Set-OrthogonalEdge 'e-ret-plus-img'
Set-OrthogonalEdge 'e-tr-plus-tok' @(@(1180, 1110), @(2079, 1110))
Set-OrthogonalEdge 'e-ret-plus-tok' @(@(1990, 843), @(1990, 1023))
Set-OrthogonalEdge 'e-ie-phie'
Set-OrthogonalEdge 'e-te-phie'
Set-OrthogonalEdge 'e-overlap-out' @(@(1965, 1125), @(3425, 1125), @(3425, 846))
Set-OrthogonalEdge 'e-resize-out'

# AttFPN detail: clean parallel branches with short right-angle joins.
Set-OrthogonalEdge 'e-stack-ch' @(@(735, 1379), @(735, 1336))
Set-OrthogonalEdge 'e-stack-sp' @(@(735, 1379), @(735, 1456))
Set-OrthogonalEdge 'e-ch-plus' @(@(1425, 1336), @(1425, 1399))
Set-OrthogonalEdge 'e-sp-plus' @(@(1425, 1456), @(1425, 1399))
Set-OrthogonalEdge 'e-base-res' @(@(1894, 1458))

# Bilayer detail: move the auxiliary mask below the residual rails, then route
# the original and returned tensors along separate non-intersecting corridors.
Set-Geometry 'd-overlap' 3185 1640 180 82
Set-Geometry 'd-note' 2680 1760 1340 28
Set-OrthogonalEdge 'e-dir-phir'
Set-OrthogonalEdge 'e-dtr-phir'
Set-OrthogonalEdge 'e-phir-overlap2' @(@(3145, 1585), @(3275, 1585))
Set-OrthogonalEdge 'e-phir-imgplus2'
Set-OrthogonalEdge 'e-ir-imgplus2' @(@(2790, 1240), @(3229, 1240))
Set-OrthogonalEdge 'e-imgplus-ie2'
Set-OrthogonalEdge 'e-phir-tokplus2'
Set-OrthogonalEdge 'e-dtr-tokplus2' @(@(2810, 1610), @(3459, 1610))
Set-OrthogonalEdge 'e-tokplus-te2'
Set-OrthogonalEdge 'e-ie-phie2'
Set-OrthogonalEdge 'e-te-phie2'

# Local center-column cleanup requested after visual review.
# One pyramid trunk replaces the two close vertical feature routes.
Set-Geometry 'bus' 1070 550 2145 27
$busCell = Get-Cell 'bus'
$busCell.SetAttribute('value', 'shared pyramid trunk · Fᵢ¹:⁵ → RPN · Fᵢ¹:⁴ → Cascade / Mask RoIAlign')
$fpnBus = Get-Cell 'e-fpn-bus'
$fpnBus.SetAttribute('value', 'Fᵢ¹:⁵')
Set-OrthogonalEdge 'e-fpn-bus' @(@(1070, 279), @(1070, 563))
$fpnRpn = Get-Cell 'e-fpn-rpn'
$fpnRpn.SetAttribute('source', 'bus')
Set-OrthogonalEdge 'e-fpn-rpn'

# Route the prompt transfer at the outside edge and along the lower empty
# corridor instead of dropping it through the decoder / residual column.
Set-OrthogonalEdge 'e-sparse-cat' @(@(3835, 665), @(3835, 1140), @(906, 1140))
Set-OrthogonalEdge 'e-tr-plus-tok' @(@(1180, 1160), @(2079, 1160))

# Split the returned dense and token tensors so each residual addition has one
# short, unambiguous input. The auxiliary mask stays a separate side output.
Set-Geometry 'returns' 1750 800 185 70
$returnsCell = Get-Cell 'returns'
$returnsCell.SetAttribute('value', 'Iᵣᵒᵘᵗ&lt;br&gt;256×32×32')
$tokenReturn = $returnsCell.CloneNode($true)
$tokenReturn.SetAttribute('id', 'token-returns')
$tokenReturn.SetAttribute('value', 'Tᵣᵒᵘᵗ&lt;br&gt;1×10×256')
$tokenGeometry = $tokenReturn.SelectSingleNode('mxGeometry')
$tokenGeometry.SetAttribute('y', '900')
[void]$root.AppendChild($tokenReturn)

$tokenReturnEdge = (Get-Cell 'e-phir-ret').CloneNode($true)
$tokenReturnEdge.SetAttribute('id', 'e-phir-token-ret')
$tokenReturnEdge.SetAttribute('source', 'phir')
$tokenReturnEdge.SetAttribute('target', 'token-returns')
[void]$root.AppendChild($tokenReturnEdge)
Set-OrthogonalEdge 'e-phir-token-ret'

$tokenResidualEdge = Get-Cell 'e-ret-plus-tok'
$tokenResidualEdge.SetAttribute('source', 'token-returns')
Set-OrthogonalEdge 'e-ret-plus-tok'
Set-OrthogonalEdge 'e-ret-plus-img'

Set-Geometry 'overlap' 1750 1040 185 88
$overlapCell = Get-Cell 'overlap'
$overlapCell.SetAttribute('value', 'm̂ᵢⱼᵉ&lt;br&gt;overlap mask&lt;br&gt;1×128×128&lt;br&gt;&lt;b&gt;auxiliary only&lt;/b&gt;')
Set-OrthogonalEdge 'e-phir-overlap'

# The nearby overlap tensor already is the output, so remove its duplicate card
# and the long return arrow across the complete SAM lane.
foreach ($removeId in @('aux-note', 'out-overlap', 'e-overlap-out')) {
    $removeCell = Get-Cell $removeId
    [void]$root.RemoveChild($removeCell)
}

# Repair the one single-waypoint residual route as two explicit coordinates.
Set-OrthogonalEdge 'e-base-res' @(@(1894, 1460), @(1894, 1468))
# Add a subtle routing note rather than another legend panel.
$legend = Get-Cell 'legend'
$legend.SetAttribute('value', 'solid = tensor · red dashed = inference control · rounded orthogonal lines use reserved routing corridors · one clear curved branch · ⊕ = residual addition · CAT = token concatenation')

$doc.mxfile.SetAttribute('modified', '2026-08-06T00:00:00.000Z')
$doc.mxfile.diagram.SetAttribute('name', 'Clean publication architecture')
[System.IO.File]::WriteAllText($Destination, $doc.OuterXml, [System.Text.UTF8Encoding]::new($false))


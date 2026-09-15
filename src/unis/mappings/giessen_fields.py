# -*- coding: utf-8 -*-
"""
Canonicalisation of `field_of_study` values from a 19th-century German
university/matriculation dataset.

Conventions
-----------
* Canonical labels are modern German, Titlecased.
* Multiple subjects are joined with "; " in the order they appear in the source
  string (e.g. "cum. et jur" -> "Kameralwissenschaft; Rechtswissenschaft").
* Confessional qualifiers on theology are preserved
  ("Katholische Theologie", "Evangelische Theologie", "Jüdische Theologie").
* Historical terms are folded into their modern equivalent:
  Jura / die Rechte / Jurisprudenz -> Rechtswissenschaft;
  Heilkunde / Arzneiwissenschaft -> Medizin;
  Thierarzneikunde -> Tierheilkunde; Baukunst -> Architektur;
  Kriegswissenschaft -> Militärwissenschaft.
* Latin forms are mapped to their German equivalents. Note "rei salt(uariae)"
  (OCR'd as "rei aalt") = forestry.
* Editorial junk in the source cell (cross-references, stray characters) is
  stripped: e.g. "Chemie. X" -> "Chemie".
"""

FIELD_OF_STUDY_MAP = {
    # ------------------------------------------------------------------
    # Theologie
    # ------------------------------------------------------------------
    "Theologie": "Theologie",
    "theol": "Theologie",
    "Theol": "Theologie",
    "9heologie": "Theologie",
    "Thologie": "Theologie",
    "Thaologie": "Theologie",
    "Theolo-_ gie": "Theologie",
    "Theolog ie": "Theologie",
    "steol": "Theologie",
    "kath. Theologie": "Katholische Theologie",
    "Kath. Theologie": "Katholische Theologie",
    "kath. Theol": "Katholische Theologie",
    "kath. theol": "Katholische Theologie",
    "kath.theol": "Katholische Theologie",
    "kath.Theologie": "Katholische Theologie",
    "kathl theol": "Katholische Theologie",
    "kat h. Theologie": "Katholische Theologie",
    "kat Theologie": "Katholische Theologie",
    "kath. The9logie": "Katholische Theologie",
    "ev. Theologie": "Evangelische Theologie",
    "jüd. Theologie": "Jüdische Theologie",

    # ------------------------------------------------------------------
    # Rechtswissenschaft
    # ------------------------------------------------------------------
    "Jura": "Rechtswissenschaft",
    "die Rechte": "Rechtswissenschaft",
    "dis Rechte": "Rechtswissenschaft",
    "die, Rechte": "Rechtswissenschaft",
    "Rechte": "Rechtswissenschaft",
    "Jurisprudenz": "Rechtswissenschaft",
    "Jurisprud": "Rechtswissenschaft",
    "Jurisprude nz": "Rechtswissenschaft",
    "Jurisprudent": "Rechtswissenschaft",
    "Jurisprudens": "Rechtswissenschaft",
    "Juri sprudenz": "Rechtswissenschaft",
    "Jurisprudenz. —": "Rechtswissenschaft",
    "Rechtswiss": "Rechtswissenschaft",
    "Rechtswissenschaft": "Rechtswissenschaft",
    "Rechtewiss": "Rechtswissenschaft",
    "Rechtwiss": "Rechtswissenschaft",
    "Rechtswies": "Rechtswissenschaft",
    "Rechtswisa": "Rechtswissenschaft",
    "Rechtswien": "Rechtswissenschaft",
    "Rechtsw": "Rechtswissenschaft",
    "Rech tsw iss": "Rechtswissenschaft",
    "jur": "Rechtswissenschaft",
    "Jur": "Rechtswissenschaft",
    "juris": "Rechtswissenschaft",
    "Juris": "Rechtswissenschaft",
    "Ju r a": "Rechtswissenschaft",
    "Jura. Gluier, Georg siehe Gliter, Georg Christian": "Rechtswissenschaft",

    # ------------------------------------------------------------------
    # Kameralwissenschaft
    # ------------------------------------------------------------------
    "Cameralwiss": "Kameralwissenschaft",
    "Camerale": "Kameralwissenschaft",
    "camerale": "Kameralwissenschaft",
    "Cameralia": "Kameralwissenschaft",
    "cameralia": "Kameralwissenschaft",
    "Cameral": "Kameralwissenschaft",
    "Cameralw": "Kameralwissenschaft",
    "Cameralwissenschaft": "Kameralwissenschaft",
    "Cameralwise": "Kameralwissenschaft",
    "Cameralwies": "Kameralwissenschaft",
    "Cameralwias": "Kameralwissenschaft",
    "Cameralwisa": "Kameralwissenschaft",
    "Cameraiwiss": "Kameralwissenschaft",
    "Cameral-. wissenschaft": "Kameralwissenschaft",
    "Cameral-Wiss": "Kameralwissenschaft",
    "Cameral_ wiss": "Kameralwissenschaft",
    "Cameralwiss.—": "Kameralwissenschaft",
    "Cameralsökonomie": "Kameralwissenschaft",
    "Carneralwiss": "Kameralwissenschaft",
    "Camoralwiss": "Kameralwissenschaft",
    "Camralwiss": "Kameralwissenschaft",
    "Cmaeralwiss": "Kameralwissenschaft",
    "Cameaawiss": "Kameralwissenschaft",
    "Camerlae": "Kameralwissenschaft",
    "Camerlalwiss": "Kameralwissenschaft",
    "Came maswiesenschaft": "Kameralwissenschaft",
    "Gameralwiss": "Kameralwissenschaft",
    "Oameralwiss": "Kameralwissenschaft",
    "Uameralwissenschaft": "Kameralwissenschaft",
    "Kameralwiss": "Kameralwissenschaft",
    "Kameral": "Kameralwissenschaft",
    "Kamerale": "Kameralwissenschaft",
    "Kameral-Wiss": "Kameralwissenschaft",
    "Kameralwias": "Kameralwissenschaft",
    "Kameralwisa": "Kameralwissenschaft",
    "Kamerlawise": "Kameralwissenschaft",
    "Kammeralwiss": "Kameralwissenschaft",
    "Kam eral wiss": "Kameralwissenschaft",
    "cam": "Kameralwissenschaft",
    "Cam": "Kameralwissenschaft",
    "cem": "Kameralwissenschaft",
    "oam": "Kameralwissenschaft",
    "rer. oam": "Kameralwissenschaft",

    # ------------------------------------------------------------------
    # Medizin, Chirurgie, Pharmazie, Tierheilkunde
    # ------------------------------------------------------------------
    "Medizin": "Medizin",
    "med": "Medizin",
    "Med": "Medizin",
    "Zed": "Medizin",  # OCR of "Med"
    "Heilkunde": "Medizin",
    "Arzneikunde": "Medizin",
    "Arzneywissenschaft": "Medizin",
    "Arznywiesenschaft": "Medizin",
    "Chirurgie": "Chirurgie",
    "Chir": "Chirurgie",
    "chir": "Chirurgie",
    "Chirur-' gie": "Chirurgie",
    "obstetret": "Geburtshilfe",
    "Pharmacie": "Pharmazie",
    "Pharmazie": "Pharmazie",
    "pharm": "Pharmazie",
    "Pharm": "Pharmazie",
    "Pharmac": "Pharmazie",
    "pharmac": "Pharmazie",
    "Pharmade": "Pharmazie",
    "Pharmscie": "Pharmazie",
    "Phamacie": "Pharmazie",
    "Pahrmacie": "Pharmazie",
    "Pharmaoie": "Pharmazie",
    "Pharmaciei": "Pharmazie",
    "Pharmäcie": "Pharmazie",
    "Pharmacie. Licentiirt": "Pharmazie",
    "Thierarzneikunde": "Tierheilkunde",
    "Thierarzneik": "Tierheilkunde",
    "Thierarznei": "Tierheilkunde",
    "Thierarzn": "Tierheilkunde",
    "Thierarznekunde": "Tierheilkunde",
    "Thierarzneikupde": "Tierheilkunde",
    "Thierarzneiwies": "Tierheilkunde",
    "Thierarzneykund": "Tierheilkunde",
    "Thierarzneykunde": "Tierheilkunde",
    "Thieraszneikunde": "Tierheilkunde",
    "Thierheilkunde": "Tierheilkunde",
    "Thierheilk": "Tierheilkunde",
    "vet.med": "Tierheilkunde",
    "vet. med": "Tierheilkunde",
    "vet.-med": "Tierheilkunde",
    "med.vet": "Tierheilkunde",
    "med. vet": "Tierheilkunde",
    "ars veterinariae": "Tierheilkunde",

    # ------------------------------------------------------------------
    # Forstwissenschaft
    # ------------------------------------------------------------------
    "Forstwiss": "Forstwissenschaft",
    "forstwiss": "Forstwissenschaft",
    "Forstwissenschaft": "Forstwissenschaft",
    "Forstwies": "Forstwissenschaft",
    "Forstwise": "Forstwissenschaft",
    "Forstwie": "Forstwissenschaft",
    "Forstw": "Forstwissenschaft",
    "Foretw": "Forstwissenschaft",
    "Foretwiss": "Forstwissenschaft",
    "Foratwiss": "Forstwissenschaft",
    "Forstiss": "Forstwissenschaft",
    "Porstwiss": "Forstwissenschaft",
    "Torstwiss": "Forstwissenschaft",
    "k'orstwiss": "Forstwissenschaft",
    "Fors twiss": "Forstwissenschaft",
    "Forst wissenschaft": "Forstwissenschaft",
    "Forstwiesenschaft": "Forstwissenschaft",
    "Forstwiesensohaft": "Forstwissenschaft",
    "Forst— ' Wissenschaft": "Forstwissenschaft",
    "Forstwiss. \\": "Forstwissenschaft",
    "Forstales": "Forstwissenschaft",
    "oeconom. forestalis": "Forstwissenschaft",
    "rei aalt": "Forstwissenschaft",  # lat. "rei saltuariae"

    # ------------------------------------------------------------------
    # Philosophische Fächer, Naturwissenschaften, Technik
    # ------------------------------------------------------------------
    "Philosophie": "Philosophie",
    "phil": "Philosophie",
    "philos": "Philosophie",
    "Philos": "Philosophie",
    "Philoso phie": "Philosophie",
    "Pholosophie": "Philosophie",
    "Philologie": "Philologie",
    "philol": "Philologie",
    "Philol": "Philologie",
    "Philolgie": "Philologie",
    "Mathematik": "Mathematik",
    "Math": "Mathematik",
    "Mathemtik": "Mathematik",
    "Mathemat_k": "Mathematik",
    "Chemie": "Chemie",
    "chem": "Chemie",
    "CHemie": "Chemie",
    "Chemie. X": "Chemie",
    "Naturwiss": "Naturwissenschaften",
    "Naturwissenschaften": "Naturwissenschaften",
    "Mineralogie": "Mineralogie",
    "Bergwissenschaft": "Bergwissenschaft",
    "Bergwiss": "Bergwissenschaft",
    "Bergbaukunde": "Bergwissenschaft",
    "Geschichte": "Geschichte",
    "Mechanik": "Mechanik",
    "Staatswissenschaft": "Staatswissenschaft",
    "Militärwiss": "Militärwissenschaft",
    "Militärische Wies": "Militärwissenschaft",
    "Kriegswissenschaft": "Militärwissenschaft",
    "Oekonomie": "Ökonomie",
    "Oeconomie": "Ökonomie",
    "Landökonomie": "Landwirtschaft",
    "Architektur": "Architektur",
    "Architekt": "Architektur",
    "Architectur": "Architektur",
    "Arohitektur": "Architektur",
    "Achitektur": "Architektur",
    "Arch itektur": "Architektur",
    "architecturae": "Architektur",
    "arch": "Architektur",
    "Baukunst": "Architektur",
    "das technische Fach": "Technik",
    "klassische u. morgenländische Sprachen":
        "Klassische Philologie; Orientalische Sprachen",

    # ------------------------------------------------------------------
    # Kombinationen: Theologie + Philologie / Philosophie
    # ------------------------------------------------------------------
    "Theologie u. Philologie": "Theologie; Philologie",
    "Theologie und Philologie": "Theologie; Philologie",
    "Theol. und Philologie": "Theologie; Philologie",
    "Theol. u. Philologie": "Theologie; Philologie",
    "Theol.u. Philologie": "Theologie; Philologie",
    "Theol. u. Philol": "Theologie; Philologie",
    "Theologie u. Philol": "Theologie; Philologie",
    "Theologie. u. Philologie": "Theologie; Philologie",
    "Theologie. u. Philol": "Theologie; Philologie",
    "theol. u. philol": "Theologie; Philologie",
    "theol. u. Philologie": "Theologie; Philologie",
    "theol. und Philologie": "Theologie; Philologie",
    "theol. u. philologie": "Theologie; Philologie",
    "theol. et philol": "Theologie; Philologie",
    "theolog. u. philolog. Wissensch": "Theologie; Philologie",
    "Philologie und Theologie": "Philologie; Theologie",
    "Philologie u. Theologie": "Philologie; Theologie",
    "Phil ologie. u. Theologie": "Philologie; Theologie",
    "philol.et theol": "Philologie; Theologie",
    "philol. et theol": "Philologie; Theologie",
    "Philosophie und Theologie": "Philosophie; Theologie",
    "phil. et theol": "Philosophie; Theologie",
    "Theologieu. Philos oph ie": "Theologie; Philosophie",
    "kath. theol. et philol": "Katholische Theologie; Philologie",
    "kath.theol. et philol": "Katholische Theologie; Philologie",
    "kath. theol. u. Philologie": "Katholische Theologie; Philologie",
    "kath. Theologie. u. Philologie": "Katholische Theologie; Philologie",
    "kath. Theol. u. Philol": "Katholische Theologie; Philologie",
    "Theologie u. orient. Sprachen": "Theologie; Orientalische Sprachen",

    # ------------------------------------------------------------------
    # Kombinationen: Recht + Kameralia u.a.
    # ------------------------------------------------------------------
    "Jura und Camerale": "Rechtswissenschaft; Kameralwissenschaft",
    "Jura u. Camerale": "Rechtswissenschaft; Kameralwissenschaft",
    "Jura u. Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Jura und Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Jura et Cameralia": "Rechtswissenschaft; Kameralwissenschaft",
    "Jura et Camerale": "Rechtswissenschaft; Kameralwissenschaft",
    "Juta und Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Jur. und Camerale": "Rechtswissenschaft; Kameralwissenschaft",
    "Jur. und Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Rechts- und Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Rechts- u. Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Rechte- u. Cameralwiss": "Rechtswissenschaft; Kameralwissenschaft",
    "Rechtswiss. u. Camerale": "Rechtswissenschaft; Kameralwissenschaft",
    "jur. et cam": "Rechtswissenschaft; Kameralwissenschaft",
    "jur. et.cam": "Rechtswissenschaft; Kameralwissenschaft",
    "jur. et cameral": "Rechtswissenschaft; Kameralwissenschaft",
    "jur. et cum": "Rechtswissenschaft; Kameralwissenschaft",
    "jur et cum": "Rechtswissenschaft; Kameralwissenschaft",
    "jur. etcum": "Rechtswissenschaft; Kameralwissenschaft",
    "Jur. et cum": "Rechtswissenschaft; Kameralwissenschaft",
    "cum. et jur": "Kameralwissenschaft; Rechtswissenschaft",
    "Cameralia u. Jura": "Kameralwissenschaft; Rechtswissenschaft",
    "Camerale und Jura": "Kameralwissenschaft; Rechtswissenschaft",
    "Camerale u. Rechtswiss": "Kameralwissenschaft; Rechtswissenschaft",
    "Kameral- u. Rechtswiss": "Kameralwissenschaft; Rechtswissenschaft",
    "Jura und Philosophie": "Rechtswissenschaft; Philosophie",
    "jur. et phil": "Rechtswissenschaft; Philosophie",
    "jur. und Mathematik": "Rechtswissenschaft; Mathematik",
    "Rechtswiss. u. kath. Theologie": "Rechtswissenschaft; Katholische Theologie",
    "kath. theol. u. Rechtswiss": "Katholische Theologie; Rechtswissenschaft",
    "Theologie u. Cameralwiss": "Theologie; Kameralwissenschaft",

    # ------------------------------------------------------------------
    # Kombinationen: Forst + Kameralia
    # ------------------------------------------------------------------
    "Forst- u. Cameralwiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forst- und Cameralwiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forst- u. Camerale": "Forstwissenschaft; Kameralwissenschaft",
    "Forst- u. Kameralwiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forst u. Kameralwiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forst- u. Camerlawiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forst- und Cameral-Wiss": "Forstwissenschaft; Kameralwissenschaft",
    "Forstwiss. u. Camerale": "Forstwissenschaft; Kameralwissenschaft",
    "Forstwiss. u. Kameralia": "Forstwissenschaft; Kameralwissenschaft",
    "Forstwissenschaft u. Cameralia": "Forstwissenschaft; Kameralwissenschaft",
    "forest et cam": "Forstwissenschaft; Kameralwissenschaft",
    "rei aalt et cam": "Forstwissenschaft; Kameralwissenschaft",
    "cam.rei. aalt": "Kameralwissenschaft; Forstwissenschaft",
    "Camerale u. Forstwiss": "Kameralwissenschaft; Forstwissenschaft",
    "Camerale und Forstwiss": "Kameralwissenschaft; Forstwissenschaft",
    "Cameral- u. Forstwiss": "Kameralwissenschaft; Forstwissenschaft",
    "Cameral- und Forstwiss": "Kameralwissenschaft; Forstwissenschaft",
    "cam. et Forstwiss": "Kameralwissenschaft; Forstwissenschaft",
    "Oeconomie und Forstwiss": "Ökonomie; Forstwissenschaft",

    # ------------------------------------------------------------------
    # Kombinationen: Kameralia/Mathematik/Ökonomie
    # ------------------------------------------------------------------
    "Camerale und Mathematik": "Kameralwissenschaft; Mathematik",
    "Camerale u. Mathematik": "Kameralwissenschaft; Mathematik",
    "cam. u. math": "Kameralwissenschaft; Mathematik",
    "Mathematik u. Camerale": "Mathematik; Kameralwissenschaft",
    "Mathematik und Camerale": "Mathematik; Kameralwissenschaft",
    "Mathematik und Cameral": "Mathematik; Kameralwissenschaft",
    "Mathematik und Cameralia": "Mathematik; Kameralwissenschaft",
    "Mathematik und Kameralia": "Mathematik; Kameralwissenschaft",
    "Mathematik u. Kameralwies": "Mathematik; Kameralwissenschaft",
    "Cameralwiss. und Oeconomie": "Kameralwissenschaft; Ökonomie",
    "Cameralwiss. u. Meßkunst": "Kameralwissenschaft; Vermessungskunde",
    "Ökonomie u. Mathematik": "Ökonomie; Mathematik",
    "Mathematik und Oeconomie": "Mathematik; Ökonomie",

    # ------------------------------------------------------------------
    # Kombinationen: Medizin, Chirurgie, Pharmazie
    # ------------------------------------------------------------------
    "Medizin und Chirurgie": "Medizin; Chirurgie",
    "Medizin u. Chirurgie": "Medizin; Chirurgie",
    "Medizin. und Chirurgie": "Medizin; Chirurgie",
    "Medizin und Chiru rgi e": "Medizin; Chirurgie",
    "med. et chir": "Medizin; Chirurgie",
    "Arzneykunst und Chirurgie": "Medizin; Chirurgie",
    "Chirurgie und Medizin": "Chirurgie; Medizin",
    "Chirurgie und Medizin. C siehe auch K": "Chirurgie; Medizin",
    "chir. et med": "Chirurgie; Medizin",
    "Medizin und Philosophie": "Medizin; Philosophie",
    "philos.Wiss. und Medizin": "Philosophie; Medizin",
    "Humaniora und Medizin": "Humaniora; Medizin",
    "Chemie und Medizin": "Chemie; Medizin",
    "Pharmazie und Medizin": "Pharmazie; Medizin",
    "Pharmacie und Chirurgie": "Pharmazie; Chirurgie",
    "Pharmacie u. Thierarzneik": "Pharmazie; Tierheilkunde",
    "Pharmacie und Thierarzneikunde": "Pharmazie; Tierheilkunde",
    "Pharmacie u. Mathematik": "Pharmazie; Mathematik",
    "Chemie u. Pharmacie": "Chemie; Pharmazie",

    # ------------------------------------------------------------------
    # Übrige Kombinationen
    # ------------------------------------------------------------------
    "Chemie u. Mathematik": "Chemie; Mathematik",
    "Mathematik u. Chemie": "Mathematik; Chemie",
    "Physik u. Mathematik": "Physik; Mathematik",
    "Physik und Chemie": "Physik; Chemie",
    "Botanik' und Chemie": "Botanik; Chemie",
    "Mathematik und Botanik": "Mathematik; Botanik",
    "Chemie und naturhietorische Gegenstände": "Chemie; Naturgeschichte",
    "deutsche Sprache und Chemie": "Deutsche Sprache; Chemie",
    "Mathematik und Philologie": "Mathematik; Philologie",
    "Philosophie und Philologie": "Philosophie; Philologie",
    "Philos. u. Philol": "Philosophie; Philologie",
    "philol. u. philos": "Philologie; Philosophie",
    "Philologieund Philosophie": "Philologie; Philosophie",
    "Philosophie u. Mathematik": "Philosophie; Mathematik",
    "Architektur und Mathematik": "Architektur; Mathematik",
    "Baukunst u. Mathematik": "Architektur; Mathematik",
    "Wasser- u. Chausseebau": "Wasserbau; Straßenbau",
}


def canonicalise(value, default="Unknown"):
    """Return the canonical label for a raw `field_of_study` value."""
    if value is None:
        return default
    return FIELD_OF_STUDY_MAP.get(value.strip(), default)
